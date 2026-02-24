"""
utils/data_loader.py

This module provides data loading utilities for the BioKG dataset:
- BioKGDataset: Handles loading triples and mappings from pickle files
- FastNegativeSampler: Efficient negative sampling with type constraints
- create_dataloader: Factory function for PyTorch DataLoader with Colab-optimized settings
"""

import numpy as np
import torch
import pickle
import os
import warnings
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# =============================================================================
# BioKG Dataset Class
# =============================================================================
class BioKGDataset(Dataset):
    """
    PyTorch Dataset for BioKG knowledge graph.
    
    Handles loading of:
    - Training/validation/test triples (from .npy or .txt files)
    - Entity and relation mappings (string ID -> integer index)
    - Type information for entities (for type-constrained sampling)
    
    The dataset uses memory mapping for large files to enable instant loading
    without consuming excessive RAM.
    """
    
    def __init__(self, split='train', data_dir='data/processed'):
        """
        Args:
            split: One of 'train', 'valid', or 'test'
            data_dir: Directory containing processed data files
        """
        self.data_dir = data_dir
        self.split = split
        
        # =====================================================================
        # Load Triples
        # =====================================================================
        # Try loading from .npy first (fastest), fall back to .txt
        triples_path = os.path.join(data_dir, f"{split}_triples.npy")
        triples_txt_path = os.path.join(data_dir, f"{split}_triples.txt")
        
        if os.path.exists(triples_path):
            # Memory map for instant loading - doesn't load into RAM until accessed
            # This is crucial for large datasets (millions of triples)
            self.triples = np.load(triples_path, mmap_mode='r')
        elif os.path.exists(triples_txt_path):
            # Load from txt and cache as npy for next time
            print(f"Loading {split} triples from txt file...")
            triples_list = []
            with open(triples_txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Convert to integers - triples files should already have integer IDs
                        triples_list.append([int(parts[0]), int(parts[1]), int(parts[2])])
            self.triples = np.array(triples_list)
            # Save as npy for faster loading next time
            np.save(triples_path, self.triples)
        else:
            raise FileNotFoundError(f"No triples file found for {split} split")
        
        # =====================================================================
        # Load Mappings (string ID -> integer index)
        # =====================================================================
        # These mappings convert the original string IDs (e.g., 'P12345') to
        # contiguous integer indices (0, 1, 2, ...) for embedding lookup
        self.entity2id = self._load_pickle(os.path.join(data_dir, "entity2id.pkl"), {})
        self.relation2id = self._load_pickle(os.path.join(data_dir, "relation2id.pkl"), {})
        
        # Create reverse mappings (id to string) for lookup when needed
        # Useful for debugging and converting predictions back to original IDs
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        # =====================================================================
        # Load Type Information (for type-constrained sampling)
        # =====================================================================
        # id_to_type maps string entity IDs to their type (e.g., 'gene', 'drug', 'disease')
        self.id_to_type = self._load_pickle(os.path.join(data_dir, "id_to_type.pkl"), None)
        
        # Build entity to type mapping using integer IDs for fast lookup
        self.entity_to_type = {}          # Maps integer entity ID -> type
        self.type_entity_indices = {}     # Maps type -> list of entity IDs (for sampling)
        
        if self.id_to_type and isinstance(self.id_to_type, dict):
            # Convert string-based mapping to integer-based mapping
            for entity_str, entity_type in self.id_to_type.items():
                if entity_str in self.entity2id:
                    entity_id = self.entity2id[entity_str]
                    self.entity_to_type[entity_id] = entity_type
            
            # Build reverse mapping (type -> list of entities) for fast sampling
            if self.entity_to_type:
                type_to_entities = {}
                for entity_id, entity_type in self.entity_to_type.items():
                    if entity_type not in type_to_entities:
                        type_to_entities[entity_type] = []
                    type_to_entities[entity_type].append(entity_id)
                
                # Convert to numpy arrays for fast sampling
                # Using numpy arrays allows for vectorized operations
                self.type_entity_indices = {
                    str(tid): np.array(entities, dtype=np.int32)
                    for tid, entities in type_to_entities.items()
                }
        
        # Dataset statistics
        self.n_entities = len(self.entity2id) if self.entity2id else 0
        self.n_relations = len(self.relation2id) if self.relation2id else 0
        
        print(f"Loaded {split} set with {len(self.triples)} triples")
        print(f"  Entities: {self.n_entities}, Relations: {self.n_relations}")
        print(f"  Entities with type info: {len(self.entity_to_type)}")
        print(f"  Entity types available: {len(self.type_entity_indices)}")
    
    def _load_pickle(self, path, default):
        """
        Safely load a pickle file with error handling.
        
        Args:
            path: Path to pickle file
            default: Default value to return if file doesn't exist or fails to load
            
        Returns:
            Unpickled object or default value
        """
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                return default
        return default
    
    def __len__(self):
        """Return number of triples in the dataset."""
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Get a single triple by index.
        
        Args:
            idx: Index of the triple
            
        Returns:
            torch.Tensor of shape (3,) with (head, relation, tail) indices
        """
        h, r, t = self.triples[idx]
        # Ensure indices are within bounds (safety check)
        h = min(max(0, int(h)), self.n_entities - 1)
        r = min(max(0, int(r)), self.n_relations - 1)
        t = min(max(0, int(t)), self.n_entities - 1)
        return torch.tensor([h, r, t], dtype=torch.long)
    
    def get_entity_type(self, entity_id):
        """
        Fast entity type lookup using integer ID.
        
        Args:
            entity_id: Integer entity ID
            
        Returns:
            Entity type string or None if not found
        """
        if self.entity_to_type is not None:
            return self.entity_to_type.get(int(entity_id), None)
        return None
    
    def get_entity_string_id(self, entity_id):
        """
        Get the original string ID for an entity (for debugging).
        
        Args:
            entity_id: Integer entity ID
            
        Returns:
            Original string ID or string representation of the integer
        """
        return self.id2entity.get(int(entity_id), str(entity_id))


# =============================================================================
# Fast Negative Sampler
# =============================================================================
class FastNegativeSampler:
    """
    Fast negative sampler with optional type constraints.
    
    Negative sampling creates corrupted triples by replacing either the head
    or tail entity with a random entity. Type-constrained sampling ensures
    the corrupted entity has the same type as the original, creating more
    realistic and challenging negatives.
    
    For example, when corrupting a drug-protein interaction, we sample from
    other drugs (not arbitrary entities) to create harder negatives.
    """
    
    def __init__(self, dataset, n_negatives=64, use_type_constraint=True, device='cpu'):
        """
        Args:
            dataset: BioKGDataset instance
            n_negatives: Number of negative samples per positive
            use_type_constraint: Whether to use type-constrained sampling
            device: Device to place tensors on (cuda/cpu)
        """
        self.dataset = dataset
        self.n_negatives = n_negatives
        self.use_type_constraint = use_type_constraint
        self.device = device
        self.n_entities = dataset.n_entities
        
        # Pre-compute type-based entity lists as tensors for fast sampling
        self.type_tensors = {}
        if use_type_constraint and hasattr(dataset, 'type_entity_indices'):
            if dataset.type_entity_indices:  # Check if not empty
                for tid, entities in dataset.type_entity_indices.items():
                    if len(entities) > 0:
                        # Convert numpy array to torch tensor and move to device
                        self.type_tensors[str(tid)] = torch.tensor(entities, device=device)
                print(f"Type-constrained sampling enabled with {len(self.type_tensors)} types")
            else:
                print("No type indices available, using uniform sampling")
        else:
            print("Type constraints disabled or not available, using uniform sampling")
    
    def sample(self, batch):
        """
        Generate negative samples for a batch of positive triples.
        
        Args:
            batch: Tensor of shape (batch_size, 3) with positive triples
            
        Returns:
            Tensor of shape (batch_size * n_negatives, 3) with negative samples
            The negatives are stacked such that the first batch_size negatives
            correspond to the first positive, etc.
        """
        if self.n_negatives == 1:
            return self._sample_single(batch)
        else:
            return self._sample_multiple(batch)
    
    def _sample_single(self, batch):
        """
        Generate ONE negative sample per positive triple.
        
        For each positive triple, we randomly decide to corrupt either the head
        or the tail, then replace it with a random entity (optionally of the same type).
        
        Args:
            batch: Tensor of shape (batch_size, 3) with positive triples
            
        Returns:
            Tensor of shape (batch_size, 3) with negative samples
        """
        device = batch.device
        batch_size = batch.shape[0]
        
        # Start with copies of the positive triples
        negative_samples = batch.clone()
        
        # Randomly decide which position to corrupt for each triple
        # 50% chance to corrupt head, 50% chance to corrupt tail
        corrupt_head = torch.rand(batch_size, device=device) < 0.5
        
        if self.use_type_constraint and self.type_tensors:
            # =============================================================
            # Type-Constrained Sampling (slower but better negatives)
            # =============================================================
            # For each triple, we need to sample from entities of the same type
            # This requires a loop since each triple may have a different type
            for i in range(batch_size):
                if corrupt_head[i]:
                    eid = batch[i, 0].item()  # Head entity
                else:
                    eid = batch[i, 2].item()  # Tail entity
                
                # Get the type of the original entity
                etype = self.dataset.get_entity_type(eid)
                
                if etype is not None and str(etype) in self.type_tensors:
                    # Sample from entities of the same type
                    type_entities = self.type_tensors[str(etype)]
                    if len(type_entities) > 0:
                        idx = torch.randint(0, len(type_entities), (1,), device=device)
                        new_entity = type_entities[idx]
                        
                        if corrupt_head[i]:
                            negative_samples[i, 0] = new_entity
                        else:
                            negative_samples[i, 2] = new_entity
                    else:
                        # Fallback to uniform if type has no entities (shouldn't happen)
                        new_entity = torch.randint(0, self.n_entities, (1,), device=device)
                        if corrupt_head[i]:
                            negative_samples[i, 0] = new_entity
                        else:
                            negative_samples[i, 2] = new_entity
                else:
                    # Fallback to uniform if entity has no type or type not in our lists
                    new_entity = torch.randint(0, self.n_entities, (1,), device=device)
                    if corrupt_head[i]:
                        negative_samples[i, 0] = new_entity
                    else:
                        negative_samples[i, 2] = new_entity
        else:
            # =============================================================
            # Uniform Sampling (faster but potentially easier negatives)
            # =============================================================
            # Fully vectorized uniform sampling - much faster but may create
            # negatives that are obviously wrong (e.g., drug as a protein)
            random_entities = torch.randint(0, max(1, self.n_entities), (batch_size,), device=device)
            
            # Replace heads where corrupt_head is True
            negative_samples[:, 0] = torch.where(corrupt_head, random_entities, negative_samples[:, 0])
            
            # Replace tails where corrupt_head is False
            negative_samples[:, 2] = torch.where(~corrupt_head, random_entities, negative_samples[:, 2])
        
        return negative_samples
    
    def _sample_multiple(self, batch):
        """
        Generate MULTIPLE negative samples per positive triple.
        
        Simply calls _sample_single n_negatives times and concatenates results.
        The output is organized as [neg for pos1, neg for pos1, ..., neg for pos2, ...]
        This ordering is important for the loss function's reshaping logic.
        
        Args:
            batch: Tensor of shape (batch_size, 3) with positive triples
            
        Returns:
            Tensor of shape (batch_size * n_negatives, 3) with negative samples
        """
        batch_size = batch.shape[0]
        
        # Generate all negatives by repeated sampling
        all_negatives = []
        for _ in range(self.n_negatives):
            neg = self._sample_single(batch)
            all_negatives.append(neg)
        
        # Concatenate along batch dimension
        # Result shape: (batch_size * n_negatives, 3)
        return torch.cat(all_negatives, dim=0)


# =============================================================================
# DataLoader Factory
# =============================================================================
def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader with Colab-optimized settings.
    
    In Google Colab, multiprocessing (num_workers > 0) can cause issues,
    so we default to 0 workers. This is slower but more stable.
    
    Args:
        dataset: BioKGDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for Colab compatibility)
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,          # 0 for Colab compatibility
        pin_memory=False,                  # Disable pin_memory in Colab
        prefetch_factor=None,               # Not used when num_workers=0
        persistent_workers=False            # Not used when num_workers=0
    )
