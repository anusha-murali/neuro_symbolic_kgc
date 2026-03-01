"""
main.py

This script implements a neuro-symbolic knowledge graph completion model for the BioKG dataset.
It combines neural embeddings (ComplEx) with biological rules mined from the graph structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import pickle
import sys
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from utils.data_loader import BioKGDataset, FastNegativeSampler, create_dataloader
from models.neuro_symbolic import NeuroSymbolicKGC
from utils.rule_miner import BiologicalRuleMiner

# Add current directory to path to import custom modules
sys.path.append('.')

# =============================================================================
# Mixed Precision Training Setup
# =============================================================================
# Mixed precision (FP16) training speeds up computation and reduces memory usage
# We provide fallbacks for environments without AMP support
try:
    from torch.cuda.amp import GradScaler, autocast
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    print("Mixed precision not available, using FP32")
    
    # Dummy classes for environments without AMP
    class autocast:
        def __init__(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class GradScaler:
        def __init__(self): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass


# =============================================================================
# Google Drive Integration Functions
# =============================================================================
def setup_google_drive(mount_drive=True):
    """
    Setup Google Drive for checkpoint saving with proper Colab handling.
    
    Args:
        mount_drive (bool): Whether to attempt mounting Google Drive
        
    Returns:
        str: Path to checkpoint directory (either local or Drive)
    """
    if mount_drive:
        try:
            from google.colab import drive
    
            drive.mount('/content/drive')
            
            # Create a dedicated directory for BioKG checkpoints in Drive
            drive_checkpoint_dir = '/content/drive/MyDrive/biokg_checkpoints'
            os.makedirs(drive_checkpoint_dir, exist_ok=True)
            
            # Create a timestamped subdirectory for this specific run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(drive_checkpoint_dir, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            return run_dir
        except ImportError:
            # Not in Colab environment, use local directory
            print("Not running in Google Colab, using local checkpoint directory")
            os.makedirs('checkpoints', exist_ok=True)
            return 'checkpoints'
    else:
        # Use local directory explicitly
        os.makedirs('checkpoints', exist_ok=True)
        return 'checkpoints'


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pt', save_to_drive=True):
    """
    Save checkpoint locally and optionally to Google Drive.
    
    Args:
        state (dict): Model state and metadata to save
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Name of checkpoint file
        save_to_drive (bool): Whether to also save to Drive
    """
    try:
        # Always save locally
        local_path = os.path.join('checkpoints', filename)
        torch.save(state, local_path)
        
        # Also save to Drive if enabled
        if save_to_drive and checkpoint_dir != 'checkpoints':
            drive_path = os.path.join(checkpoint_dir, filename)
            torch.save(state, drive_path)
            
        # Keep a latest copy for easy resume
        latest_local = os.path.join('checkpoints', 'latest.pt')
        torch.save(state, latest_local)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


# =============================================================================
# Configuration and Validation Functions
# =============================================================================
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_full(model, val_loader, device, config, hr_to_tails):
    """
    Full validation loop with Filtered metrics (standard KGC evaluation).
    
    This implements the filtered setting where known true triples are excluded
    during ranking to avoid false negatives. For each (head, relation, tail) triple,
    we compute scores for all possible tails, mask out other known true tails,
    and find the rank of the correct tail.
    
    Args:
        model: The neuro-symbolic model
        val_loader: DataLoader for validation set
        device: torch device (cuda/cpu)
        config: Configuration dictionary
        hr_to_tails: Dictionary mapping (head, relation) to list of true tails
                    (used for filtering)
    
    Returns:
        dict: Metrics including MRR, Hits@1, Hits@3, Hits@10
    """
    model.eval()
    all_ranks = []
    
    pbar = tqdm(val_loader, desc="Evaluating (Filtered)", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            # Handle potential tuple output from DataLoader
            if isinstance(batch, (list, tuple)):
                triples = batch[0].to(device)
            else:
                triples = batch.to(device)
                
            batch_size = triples.shape[0]
            
            # =================================================================
            # FILTERING STEP: Create mask to exclude known true triples
            # =================================================================
            filter_mask = torch.zeros((batch_size, model.n_entities), dtype=torch.bool, device=device)
            
            # Populate the mask with known true tails for each (h,r) pair
            for i in range(batch_size):
                h = triples[i, 0].item()
                r = triples[i, 1].item()
                target_t = triples[i, 2].item()
                
                # Get all known true tails for this (head, relation) pair from training/val/test
                known_tails = hr_to_tails.get((h, r), [])
                
                if known_tails:
                    filter_mask[i, known_tails] = True
                
                # UNMASK the target tail for this specific batch item
                # We want to evaluate this triple, so we need its score to be considered
                filter_mask[i, target_t] = False
                
            # =================================================================
            # RANKING STEP: Get model predictions and compute ranks
            # =================================================================
            # Get ranks from the model (passing filter_mask to exclude known triples)
            ranks = model.evaluate_ranks(triples, filter_mask)
            all_ranks.extend(ranks.cpu().tolist())
            
            # Update progress bar with current MRR
            current_mrr = (1.0 / torch.tensor(all_ranks, dtype=torch.float)).mean().item()
            pbar.set_postfix({'MRR': f'{current_mrr:.4f}'})
            
    # Calculate final metrics
    ranks_tensor = torch.tensor(all_ranks, dtype=torch.float)
    metrics = {
        'MRR': (1.0 / ranks_tensor).mean().item(),
        'Hits@1': (ranks_tensor <= 1).float().mean().item(),
        'Hits@3': (ranks_tensor <= 3).float().mean().item(),
        'Hits@10': (ranks_tensor <= 10).float().mean().item(),
        'num_evaluated': len(all_ranks)
    }
    
    return metrics


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Neuro-Symbolic KGC on BioKG')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--no_drive', action='store_true', help='Disable Google Drive mounting')
    args = parser.parse_args()

    # =========================================================================
    # Setup: Configuration, Device, Checkpoint Directory
    # =========================================================================
    config = load_config(args.config)
    checkpoint_dir = setup_google_drive(not args.no_drive)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # =========================================================================
    # Data Loading
    # =========================================================================
    print("\nLoading datasets...")
    train_dataset = BioKGDataset(split='train', data_dir='data/processed')
    val_dataset = BioKGDataset(split='valid', data_dir='data/processed')
    
    try:
        test_dataset = BioKGDataset(split='test', data_dir='data/processed')
    except Exception:
        test_dataset = None
        print("No test dataset found. Skipping test load.")

    # =========================================================================
    # Build Global Truth Dictionary for Filtered Evaluation
    # =========================================================================
    # This dictionary maps (head, relation) pairs to all known true tails
    # across train/val/test sets. Used during validation to mask out other
    # valid answers when computing ranks (prevents false negatives).
    print("\nBuilding global truth dictionary for filtered evaluation...")
    arrays_to_stack = [train_dataset.triples, val_dataset.triples]
    
    if test_dataset is not None:
        arrays_to_stack.append(test_dataset.triples)
        
    all_triples = np.vstack(arrays_to_stack)
    
    hr_to_tails = defaultdict(list)
    for h, r, t in all_triples:
        hr_to_tails[(int(h), int(r))].append(int(t))
        
    print(f"Global truth dictionary built with {len(all_triples)} total facts.")

    # =========================================================================
    # Initialize Negative Sampler
    # =========================================================================
    # Negative sampling creates corrupted triples for training
    # The FastNegativeSampler uses type constraints for more realistic negatives
    negative_sampler = FastNegativeSampler(
        train_dataset, 
        n_negatives=config['training']['n_negatives'], 
        device=device
    )

    # =========================================================================
    # Initialize Model, Optimizer, and Scaler
    # =========================================================================
    model = NeuroSymbolicKGC(
        n_entities=train_dataset.n_entities,
        n_relations=train_dataset.n_relations,
        embedding_dim=config['model']['embedding_dim'],
        lambda_logic=config['model'].get('lambda_logic', 0.1),
        temperature=config['model'].get('temperature', 1.0)
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler() if MIXED_PRECISION_AVAILABLE else None

    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=config['training']['batch_size'] * 2,  # Larger batch for faster validation
        shuffle=False
    )

    # =========================================================================
    # Training Loop Setup
    # =========================================================================
    best_mrr = 0.0
    patience_counter = 0
    patience_limit = config['training'].get('patience', 20)
    
    print("\n" + "=" * 60)
    print("Starting Training Loop")
    print("=" * 60)

    # =========================================================================
    # Main Training Loop
    # =========================================================================
    for epoch in range(config['training']['n_epochs']):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['n_epochs']}")
        
        for batch in pbar:
            # Move batch to GPU
            batch = batch.to(device, non_blocking=True)
            
            # Generate negative samples (corrupt heads or tails)
            neg_batch = negative_sampler.sample(batch)
            
            # Zero gradients before backward pass
            optimizer.zero_grad()
            
            # =================================================================
            # Forward and Backward Pass with Optional Mixed Precision
            # =================================================================
            if scaler and MIXED_PRECISION_AVAILABLE:
                # Mixed precision forward pass
                with autocast():
                    pos_scores, _, _ = model(batch)
                    neg_scores, _, _ = model(neg_batch)
                    
                    # Core Loss (margin-based ranking loss)
                    base_loss = model.compute_loss(pos_scores, neg_scores)
                    
                    # N3 Regularization (prevents embedding norms from growing too large)
                    # This is a common regularization for KGE models
                    reg_loss = 0.0001 * model.get_regularization(batch)
                    loss = base_loss + reg_loss
                    
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Full precision forward/backward
                pos_scores, _, _ = model(batch)
                neg_scores, _, _ = model(neg_batch)
                
                base_loss = model.compute_loss(pos_scores, neg_scores)
                reg_loss = 0.05 * model.get_regularization(batch)
                loss = base_loss + reg_loss
                
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        # End of epoch statistics
        avg_loss = total_loss / len(train_loader)
        train_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {avg_loss:.4f} | Time: {train_time:.2f}s")
        
        # =====================================================================
        # Validation (every eval_every epochs)
        # =====================================================================
        if (epoch + 1) % config['training']['eval_every'] == 0:
            val_start = time.time()
            
            # Run filtered evaluation (standard KGC evaluation protocol)
            metrics = validate_full(model, val_loader, device, config, hr_to_tails)
            
            val_time = time.time() - val_start
            
            print(f"Validation (Filtered):")
            print(f"  MRR: {metrics['MRR']:.4f}")
            print(f"  Hits@1: {metrics['Hits@1']:.4f}")
            print(f"  Hits@3: {metrics['Hits@3']:.4f}")
            print(f"  Hits@10: {metrics['Hits@10']:.4f}")
            print(f"  Time: {val_time:.2f}s")
            
            # =================================================================
            # Checkpoint Saving and Early Stopping
            # =================================================================
            if metrics['MRR'] > best_mrr:
                # New best model found - save it
                best_mrr = metrics['MRR']
                patience_counter = 0
                
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mrr': best_mrr,
                    'metrics': metrics,
                    'config': config
                }
                save_checkpoint(state, checkpoint_dir, filename='best_model.pt', save_to_drive=not args.no_drive)
                print(f"New best MRR! Checkpoint saved.")
            else:
                # No improvement - increment patience counter
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience_limit}")
                
                if patience_counter >= patience_limit:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in MRR.")
                    break

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best MRR achieved: {best_mrr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
