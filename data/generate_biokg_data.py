"""
generate_biokg_data.py
Modified to generate entities.txt with actual names from property files
"""

import os
import pickle
import numpy as np
from collections import defaultdict
import glob
from tqdm import tqdm
import pandas as pd

# Define all entity types in BioKG
ENTITY_TYPES = [
    'disease', 'drug', 'protein', 'genetic_disorder', 
    'cell', 'pathway'
]

# Define property files and their corresponding entity types
PROPERTY_FILES = {
    'biokg.properties.disease.tsv': 'disease',
    'biokg.properties.drug.tsv': 'drug',
    'biokg.properties.protein.tsv': 'protein',
    'biokg.properties.genetic_disorder.tsv': 'genetic_disorder',
    'biokg.properties.cell.tsv': 'cell',
    'biokg.properties.pathway.tsv': 'pathway'
}

# Define metadata files and their corresponding entity types
METADATA_FILES = {
    'biokg.metadata.disease.tsv': 'disease',
    'biokg.metadata.drug.tsv': 'drug',
    'biokg.metadata.pathway.tsv': 'pathway',
    'biokg.metadata.protein.tsv': 'protein'
}

# Property types that can contain entity names
NAME_PROPERTIES = ['NAME', 'FULL_NAME', 'SHORT_NAME', 'SYNONYM']

class BioKGDataProcessor:
    def __init__(self, data_dir="raw", processed_dir="processed"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        
        # Initialize data structures
        self.entities = set()  # Set of all entity IDs (type_id format)
        self.entity2id = {}  # Entity ID -> integer ID
        self.id2entity = {}  # integer ID -> Entity ID
        self.entity2name = {}  # Entity ID -> human-readable name
        self.entity_type = {}  # Entity ID -> entity type
        self.entity_properties = defaultdict(dict)  # Entity ID -> properties
        self.entity_metadata = defaultdict(dict)  # Entity ID -> metadata
        
        self.relations = set()  # Set of all relations
        self.relation2id = {}  # Relation name -> ID
        self.id2relation = {}  # ID -> Relation name
        
        self.triples = []  # List of all triples (head, relation, tail)
        
        # Entity type groupings
        self.entities_by_type = defaultdict(list)  # Entity type -> list of entity IDs
        
        # Track statistics
        self.stats = defaultdict(int)
        
        # Create output directory
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def extract_entity_name(self, entity_id, etype, properties):
        """
        Extract human-readable name from properties
        
        Args:
            entity_id: The entity ID (e.g., 'A0A075B6P5')
            etype: Entity type (e.g., 'protein')
            properties: Dictionary of properties for this entity
        
        Returns:
            Human-readable name or fallback to type_id format
        """
        # Try to find a name in properties
        for name_prop in NAME_PROPERTIES:
            if name_prop in properties:
                return properties[name_prop]
        
        # Try to find in metadata
        if entity_id in self.entity_metadata:
            for name_prop in NAME_PROPERTIES:
                if name_prop in self.entity_metadata[entity_id]:
                    return self.entity_metadata[entity_id][name_prop]
        
        # Fallback to type_id format
        return f"{etype}_{entity_id}"
    
    def load_all_entities_from_properties(self):
        """Load all entities from property files and extract names"""
        print("\n" + "=" * 60)
        print("Loading entities from property files...")
        print("=" * 60)
        
        for filename, etype in PROPERTY_FILES.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filename} not found, skipping")
                continue
            
            print(f"\nProcessing properties for {etype} from {filename}...")
            
            # Read TSV file
            df = pd.read_csv(filepath, sep='\t', header=None,
                            names=['entity_id', 'property_type', 'property_value'],
                            dtype=str)
            
            # First pass: collect all properties for each entity
            entity_props = defaultdict(dict)
            for _, row in tqdm(df.iterrows(), desc=f"Collecting {etype} properties", total=len(df)):
                entity_id = row['entity_id']
                prop_type = row['property_type']
                prop_value = row['property_value']
                
                entity_props[entity_id][prop_type] = prop_value
                
                # Extract relation from property type (some properties might be relations)
                if prop_type not in self.relations:
                    self.relations.add(prop_type)
            
            # Second pass: create entities with names
            for entity_id, props in tqdm(entity_props.items(), desc=f"Creating {etype} entities"):
                full_entity_name = f"{etype}_{entity_id}"
                
                # Add to entities set
                self.entities.add(full_entity_name)
                
                # Store entity type
                self.entity_type[full_entity_name] = etype
                
                # Store all properties
                self.entity_properties[full_entity_name].update(props)
                
                # Extract and store human-readable name
                entity_name = self.extract_entity_name(entity_id, etype, props)
                self.entity2name[full_entity_name] = entity_name
            
            print(f"  Processed {len(entity_props)} {etype} entities")
        
        print(f"\nTotal entities from properties: {len(self.entities)}")
    
    def load_all_entities_from_metadata(self):
        """Load additional entities from metadata files and extract names"""
        print("\n" + "=" * 60)
        print("Loading entities from metadata files...")
        print("=" * 60)
        
        for filename, etype in METADATA_FILES.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filename} not found, skipping")
                continue
            
            print(f"\nProcessing metadata for {etype} from {filename}...")
            
            # Read TSV file
            df = pd.read_csv(filepath, sep='\t', header=None,
                            names=['entity_id', 'metadata_type', 'metadata_value'],
                            dtype=str)
            
            # First pass: collect all metadata for each entity
            entity_meta = defaultdict(dict)
            for _, row in tqdm(df.iterrows(), desc=f"Collecting {etype} metadata", total=len(df)):
                entity_id = row['entity_id']
                meta_type = row['metadata_type']
                meta_value = row['metadata_value']
                
                entity_meta[entity_id][meta_type] = meta_value
            
            # Second pass: create entities with names
            for entity_id, meta in tqdm(entity_meta.items(), desc=f"Creating {etype} entities"):
                full_entity_name = f"{etype}_{entity_id}"
                
                # Add to entities set
                self.entities.add(full_entity_name)
                
                # Store entity type if not already set
                if full_entity_name not in self.entity_type:
                    self.entity_type[full_entity_name] = etype
                
                # Store metadata
                self.entity_metadata[full_entity_name].update(meta)
                
                # If entity already has properties, update name if we find a better one
                if full_entity_name in self.entity_properties:
                    current_name = self.entity2name.get(full_entity_name)
                    if current_name and current_name.startswith(f"{etype}_"):
                        # Current name is the fallback, try to get a better name from metadata
                        better_name = self.extract_entity_name(entity_id, etype, {})
                        if not better_name.startswith(f"{etype}_"):
                            self.entity2name[full_entity_name] = better_name
                else:
                    # New entity, extract name from metadata
                    entity_name = self.extract_entity_name(entity_id, etype, {})
                    self.entity2name[full_entity_name] = entity_name
            
            print(f"  Processed {len(entity_meta)} {etype} entities")
        
        print(f"\nTotal entities from metadata: {len(self.entities)}")
    
    def load_links(self):
        """Load all links/triples from biokg.links.tsv and extract relations"""
        print("\n" + "=" * 60)
        print("Loading links from biokg.links.tsv...")
        print("=" * 60)
        
        links_file = os.path.join(self.data_dir, "biokg.links.tsv")
        
        if not os.path.exists(links_file):
            raise FileNotFoundError(f"Links file not found: {links_file}")
        
        # Count total lines for progress bar
        with open(links_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"Processing {total_lines} links...")
        
        # Process links file
        with open(links_file, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Processing links"):
                parts = line.strip().split('\t')
                
                if len(parts) < 3:
                    continue
                
                h, r, t = parts[:3]
                
                # Add entities to set
                self.entities.add(h)
                self.entities.add(t)
                
                # Add relation to set
                self.relations.add(r)
                
                # Store triple
                self.triples.append((h, r, t))
        
        print(f"\nTotal triples loaded: {len(self.triples)}")
        print(f"Total relations from links: {len(self.relations)}")
    
    def create_mappings(self):
        """Create ID mappings for entities and relations"""
        print("\n" + "=" * 60)
        print("Creating ID mappings...")
        print("=" * 60)
        
        # Sort entities for consistent ordering
        sorted_entities = sorted(list(self.entities))
        
        # Create entity mappings
        self.entity2id = {entity: idx for idx, entity in enumerate(sorted_entities)}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        
        print(f"Mapped {len(self.entity2id)} entities")
        
        # Sort relations for consistent ordering
        sorted_relations = sorted(list(self.relations))
        
        # Create relation mappings
        self.relation2id = {rel: idx for idx, rel in enumerate(sorted_relations)}
        self.id2relation = {idx: rel for rel, idx in self.relation2id.items()}
        
        print(f"Mapped {len(self.relation2id)} relations")
    
    def create_entity_type_groups(self):
        """Group entities by their type"""
        print("\n" + "=" * 60)
        print("Grouping entities by type...")
        print("=" * 60)
        
        for entity in self.entities:
            # Extract type from entity name (format: type_id)
            etype = entity.split('_')[0]
            self.entities_by_type[etype].append(entity)
        
        print("\nEntity type distribution:")
        for etype, entities in self.entities_by_type.items():
            print(f"  {etype}: {len(entities)} entities")
    
    def create_type_to_ids_mapping(self):
        """Create type to IDs mapping for efficient sampling"""
        print("\n" + "=" * 60)
        print("Creating type to IDs mapping...")
        print("=" * 60)
        
        self.type_to_ids = {}
        self.id_to_type = {}
        
        for etype, entities in self.entities_by_type.items():
            self.type_to_ids[etype] = [self.entity2id[e] for e in entities]
            print(f"  {etype}: {len(self.type_to_ids[etype])} IDs mapped")
        
        # Create entity ID to type mapping
        for entity_id, entity_name in self.id2entity.items():
            etype = entity_name.split('_')[0]
            self.id_to_type[entity_id] = etype
    
    def create_train_valid_test_splits(self, train_ratio=0.8, valid_ratio=0.1):
        """Create train/valid/test splits from triples"""
        print("\n" + "=" * 60)
        print("Creating train/valid/test splits...")
        print("=" * 60)
        
        # Shuffle triples deterministically
        np.random.seed(42)
        indices = np.random.permutation(len(self.triples))
        
        n_train = int(len(self.triples) * train_ratio)
        n_valid = int(len(self.triples) * valid_ratio)
        
        train_indices = indices[:n_train]
        valid_indices = indices[n_train:n_train + n_valid]
        test_indices = indices[n_train + n_valid:]
        
        # Convert triples to IDs
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        
        for idx in train_indices:
            h, r, t = self.triples[idx]
            if h in self.entity2id and r in self.relation2id and t in self.entity2id:
                self.train_triples.append([self.entity2id[h], self.relation2id[r], self.entity2id[t]])
        
        for idx in valid_indices:
            h, r, t = self.triples[idx]
            if h in self.entity2id and r in self.relation2id and t in self.entity2id:
                self.valid_triples.append([self.entity2id[h], self.relation2id[r], self.entity2id[t]])
        
        for idx in test_indices:
            h, r, t = self.triples[idx]
            if h in self.entity2id and r in self.relation2id and t in self.entity2id:
                self.test_triples.append([self.entity2id[h], self.relation2id[r], self.entity2id[t]])
        
        print(f"  Train: {len(self.train_triples)} triples")
        print(f"  Valid: {len(self.valid_triples)} triples")
        print(f"  Test: {len(self.test_triples)} triples")
    
    def save_text_files(self):
        """Save entities.txt and relations.txt files with human-readable names"""
        print("\n" + "=" * 60)
        print("Saving text files with human-readable names...")
        print("=" * 60)
        
        # Save entities.txt with human-readable names
        entities_file = os.path.join(self.data_dir, "entities.txt")
        with open(entities_file, 'w', encoding='utf-8') as f:
            for entity in sorted(self.entities):
                # Get human-readable name if available, otherwise use entity ID
                entity_name = self.entity2name.get(entity, entity)
                # Escape any special characters if needed
                entity_name = entity_name.replace('\n', ' ').replace('\r', ' ').strip()
                f.write(f"{entity_name}\n")
        print(f"  Saved {entities_file} with {len(self.entities)} entities")
        
        # Save entity_id to name mapping for reference
        id_to_name_file = os.path.join(self.processed_dir, "entity_id_to_name.pkl")
        with open(id_to_name_file, 'wb') as f:
            pickle.dump(self.entity2name, f)
        print(f"  Saved {id_to_name_file} with ID to name mappings")
        
        # Save relations.txt
        relations_file = os.path.join(self.data_dir, "relations.txt")
        with open(relations_file, 'w') as f:
            for relation in sorted(self.relations):
                f.write(f"{relation}\n")
        print(f"  Saved {relations_file} with {len(self.relations)} relations")
        
        # Save train/valid/test text files with human-readable names
        splits = [
            ('train', self.train_triples),
            ('valid', self.valid_triples),
            ('test', self.test_triples)
        ]
        
        for split_name, triples in splits:
            if triples:
                split_file = os.path.join(self.data_dir, f"{split_name}.txt")
                with open(split_file, 'w', encoding='utf-8') as f:
                    for h_id, r_id, t_id in triples:
                        h = self.id2entity[h_id]
                        r = self.id2relation[r_id]
                        t = self.id2entity[t_id]
                        
                        # Get human-readable names
                        h_name = self.entity2name.get(h, h)
                        t_name = self.entity2name.get(t, t)
                        
                        # Escape any special characters
                        h_name = h_name.replace('\n', ' ').replace('\r', ' ').strip()
                        t_name = t_name.replace('\n', ' ').replace('\r', ' ').strip()
                        
                        f.write(f"{h_name}\t{r}\t{t_name}\n")
                print(f"  Saved {split_file} with {len(triples)} triples")
    
    def save_pickle_files(self):
        """Save all processed data to pickle files"""
        print("\n" + "=" * 60)
        print("Saving pickle files...")
        print("=" * 60)
        
        # Save entity mappings
        with open(os.path.join(self.processed_dir, "entity2id.pkl"), 'wb') as f:
            pickle.dump(self.entity2id, f)
        print("  Saved entity2id.pkl")
        
        with open(os.path.join(self.processed_dir, "id2entity.pkl"), 'wb') as f:
            pickle.dump(self.id2entity, f)
        print("  Saved id2entity.pkl")
        
        # Save entity name mappings
        with open(os.path.join(self.processed_dir, "entity2name.pkl"), 'wb') as f:
            pickle.dump(self.entity2name, f)
        print("  Saved entity2name.pkl")
        
        # Save relation mappings
        with open(os.path.join(self.processed_dir, "relation2id.pkl"), 'wb') as f:
            pickle.dump(self.relation2id, f)
        print("  Saved relation2id.pkl")
        
        with open(os.path.join(self.processed_dir, "id2relation.pkl"), 'wb') as f:
            pickle.dump(self.id2relation, f)
        print("  Saved id2relation.pkl")
        
        # Save entity type mappings
        with open(os.path.join(self.processed_dir, "entities_by_type.pkl"), 'wb') as f:
            pickle.dump(dict(self.entities_by_type), f)
        print("  Saved entities_by_type.pkl")
        
        with open(os.path.join(self.processed_dir, "type_to_ids.pkl"), 'wb') as f:
            pickle.dump(self.type_to_ids, f)
        print("  Saved type_to_ids.pkl")
        
        with open(os.path.join(self.processed_dir, "id_to_type.pkl"), 'wb') as f:
            pickle.dump(self.id_to_type, f)
        print("  Saved id_to_type.pkl")
        
        # Save entity properties and metadata
        with open(os.path.join(self.processed_dir, "entity_properties.pkl"), 'wb') as f:
            pickle.dump(dict(self.entity_properties), f)
        print("  Saved entity_properties.pkl")
        
        with open(os.path.join(self.processed_dir, "entity_metadata.pkl"), 'wb') as f:
            pickle.dump(dict(self.entity_metadata), f)
        print("  Saved entity_metadata.pkl")
        
        # Save dataset statistics
        stats = {
            'n_entities': len(self.entity2id),
            'n_relations': len(self.relation2id),
            'entity_types': {etype: len(entities) for etype, entities in self.entities_by_type.items()},
            'n_train': len(self.train_triples),
            'n_valid': len(self.valid_triples),
            'n_test': len(self.test_triples)
        }
        
        with open(os.path.join(self.processed_dir, "stats.pkl"), 'wb') as f:
            pickle.dump(dict(stats), f)
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def save_numpy_files(self):
        """Save triples as numpy files"""
        print("\n" + "=" * 60)
        print("Saving numpy files...")
        print("=" * 60)
        
        # Save train triples
        if self.train_triples:
            train_array = np.array(self.train_triples, dtype=np.int32)
            np.save(os.path.join(self.processed_dir, "train_triples.npy"), train_array)
            print(f"  Saved train_triples.npy with shape {train_array.shape}")
        
        # Save valid triples
        if self.valid_triples:
            valid_array = np.array(self.valid_triples, dtype=np.int32)
            np.save(os.path.join(self.processed_dir, "valid_triples.npy"), valid_array)
            print(f"  Saved valid_triples.npy with shape {valid_array.shape}")
        
        # Save test triples
        if self.test_triples:
            test_array = np.array(self.test_triples, dtype=np.int32)
            np.save(os.path.join(self.processed_dir, "test_triples.npy"), test_array)
            print(f"  Saved test_triples.npy with shape {test_array.shape}")
    
    


    def create_all_rule_files(self):
        """Create all biological rule files for BioKG"""
        print("\n" + "=" * 60)
        print("Creating all biological rule files...")
        print("=" * 60)
        
        # Create different categories of rules
        self.create_biological_rules()  # Original method for biological_rules.pkl
        
        # Create additional rule files
        self.create_symmetric_rules()
        self.create_inverse_rules()
        self.create_composition_rules()
        self.create_hierarchy_rules()
        self.create_domain_specific_rules()
        self.create_pathway_rules()
        self.create_disease_rules()
        self.create_drug_rules()
        self.create_protein_rules()
        
        # Create combined rules file
        self.create_combined_rules()
        
        print("\n" + "=" * 60)
        print("✅ All rule files created successfully!")
        print("=" * 60)

    def create_symmetric_rules(self):
        """Create symmetric relation rules"""
        print("\nCreating symmetric rules...")
        
        # Identify symmetric relations in BioKG
        symmetric_relations = [
            'PPI',  # Protein-Protein Interaction
            'INTERACTS_WITH',
            'ASSOCIATED_WITH',
            'CHEMICAL_ASSOCIATION',
            'BINDS_TO',
            'INTERACTS'
        ]
        
        symmetric_rules = []
        
        for rel in symmetric_relations:
            if rel in self.relation2id:
                rule = {
                    'name': f'{rel.lower()}_symmetric',
                    'type': 'symmetric',
                    'relation': rel,
                    'body': [(None, rel, None)],  # Any entity types
                    'head': (None, rel, None),
                    'confidence': 0.95,
                    'description': f'{rel} is a symmetric relation'
                }
                symmetric_rules.append(rule)
        
        # Save symmetric rules
        rules_path = os.path.join(self.processed_dir, "symmetric_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(symmetric_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "symmetric_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in symmetric_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Relation: {rule['relation']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(symmetric_rules)} symmetric rules")

    def create_inverse_rules(self):
        """Create inverse relation rules"""
        print("\nCreating inverse rules...")
        
        # Define inverse relation pairs in BioKG
        inverse_pairs = [
            ('DRUG_INDICATION_ASSOCIATION', 'inverse_DRUG_INDICATION_ASSOCIATION'),
            ('DRUG_SIDEEFFECT_ASSOCIATION', 'inverse_DRUG_SIDEEFFECT_ASSOCIATION'),
            ('DIRECT_PARENT', 'ALTERNATIVE_PARENT'),
            ('DISEASE_GENETIC_DISORDER', 'associated_with'),
            ('DISEASE_PATHWAY_ASSOCIATION', 'implicated_in'),
            ('HAS_PARENT_PATHWAY', 'HAS_CHILD_PATHWAY'),
            ('PROTEIN_DISEASE_ASSOCIATION', 'disease_associated_protein'),
            ('DRUG_TARGET', 'targeted_by_drug'),
            ('MEMBER_OF_COMPLEX', 'has_member'),
            ('COMPLEX_IN_PATHWAY', 'pathway_contains_complex'),
            ('ENCODES', 'encoded_by'),
            ('REGULATES', 'regulated_by'),
            ('INHIBITS', 'inhibited_by'),
            ('ACTIVATES', 'activated_by')
        ]
        
        inverse_rules = []
        
        for rel1, rel2 in inverse_pairs:
            if rel1 in self.relation2id and rel2 in self.relation2id:
                # Rule 1: rel1 -> rel2
                rule1 = {
                    'name': f'{rel1.lower()}_to_{rel2.lower()}',
                    'type': 'inverse',
                    'relation1': rel1,
                    'relation2': rel2,
                    'body': [(None, rel1, None)],
                    'head': (None, rel2, None),
                    'confidence': 0.9,
                    'description': f'{rel1} implies {rel2}'
                }
                inverse_rules.append(rule1)
                
                # Rule 2: rel2 -> rel1
                rule2 = {
                    'name': f'{rel2.lower()}_to_{rel1.lower()}',
                    'type': 'inverse',
                    'relation1': rel2,
                    'relation2': rel1,
                    'body': [(None, rel2, None)],
                    'head': (None, rel1, None),
                    'confidence': 0.9,
                    'description': f'{rel2} implies {rel1}'
                }
                inverse_rules.append(rule2)
        
        # Save inverse rules
        rules_path = os.path.join(self.processed_dir, "inverse_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(inverse_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "inverse_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in inverse_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"From: {rule['relation1']} -> To: {rule['relation2']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(inverse_rules)} inverse rules")

    def create_composition_rules(self):
        """Create composition/chain rules"""
        print("\nCreating composition rules...")
        
        # Define composition rules with entity type constraints
        composition_rules = [
            # Drug -> Protein -> Disease chains
            {
                'name': 'drug_side_effect_via_protein',
                'body': [
                    ('drug', 'DRUG_TARGET', 'protein'),
                    ('protein', 'PROTEIN_DISEASE_ASSOCIATION', 'disease')
                ],
                'head': ('drug', 'DRUG_SIDEEFFECT_ASSOCIATION', 'disease'),
                'confidence': 0.7,
                'description': 'Drug side effects can be mediated through protein targets'
            },
            {
                'name': 'drug_indication_via_pathway',
                'body': [
                    ('drug', 'DRUG_TARGET', 'protein'),
                    ('protein', 'PROTEIN_PATHWAY_ASSOCIATION', 'pathway'),
                    ('pathway', 'DISEASE_PATHWAY_ASSOCIATION', 'disease')
                ],
                'head': ('drug', 'DRUG_INDICATION_ASSOCIATION', 'disease'),
                'confidence': 0.65,
                'description': 'Drug indications can be inferred through pathway involvement'
            },
            
            # Disease -> Gene -> Protein chains
            {
                'name': 'disease_protein_via_gene',
                'body': [
                    ('disease', 'DISEASE_GENETIC_DISORDER', 'genetic_disorder'),
                    ('genetic_disorder', 'ENCODES', 'protein')
                ],
                'head': ('disease', 'PROTEIN_DISEASE_ASSOCIATION', 'protein'),
                'confidence': 0.8,
                'description': 'Disease-associated genes encode proteins involved in disease'
            },
            
            # Protein -> Complex -> Pathway chains
            {
                'name': 'protein_pathway_via_complex',
                'body': [
                    ('protein', 'MEMBER_OF_COMPLEX', 'complex'),
                    ('complex', 'COMPLEX_IN_PATHWAY', 'pathway')
                ],
                'head': ('protein', 'PROTEIN_PATHWAY_ASSOCIATION', 'pathway'),
                'confidence': 0.75,
                'description': 'Proteins in complexes participate in pathways'
            },
            
            # Pathway hierarchy chains
            {
                'name': 'pathway_grandparent',
                'body': [
                    ('pathway', 'HAS_PARENT_PATHWAY', 'pathway2'),
                    ('pathway2', 'HAS_PARENT_PATHWAY', 'pathway3')
                ],
                'head': ('pathway', 'HAS_PARENT_PATHWAY', 'pathway3'),
                'confidence': 0.9,
                'description': 'Transitive hierarchy in pathways'
            },
            
            # Disease hierarchy chains
            {
                'name': 'disease_grandparent',
                'body': [
                    ('disease', 'DIRECT_PARENT', 'disease2'),
                    ('disease2', 'DIRECT_PARENT', 'disease3')
                ],
                'head': ('disease', 'DIRECT_PARENT', 'disease3'),
                'confidence': 0.95,
                'description': 'Transitive hierarchy in diseases'
            },
            
            # Drug similarity chains
            {
                'name': 'drug_similarity_transitive',
                'body': [
                    ('drug', 'CHEMICAL_SIMILARITY', 'drug2'),
                    ('drug2', 'CHEMICAL_SIMILARITY', 'drug3')
                ],
                'head': ('drug', 'CHEMICAL_SIMILARITY', 'drug3'),
                'confidence': 0.85,
                'description': 'Transitive chemical similarity'
            },
            
            # Protein interaction chains
            {
                'name': 'protein_interaction_chain',
                'body': [
                    ('protein', 'PPI', 'protein2'),
                    ('protein2', 'PPI', 'protein3')
                ],
                'head': ('protein', 'PPI', 'protein3'),
                'confidence': 0.6,
                'description': 'Transitive protein-protein interactions (with lower confidence)'
            }
        ]
        
        # Filter rules based on available relations
        available_relations = set(self.relation2id.keys())
        valid_rules = []
        
        for rule in composition_rules:
            # Collect all relations in this rule
            relations_in_rule = set()
            
            # Add body relations
            for body_elem in rule['body']:
                if len(body_elem) >= 2:
                    relations_in_rule.add(body_elem[1])
            
            # Add head relation
            if len(rule['head']) >= 2:
                relations_in_rule.add(rule['head'][1])
            
            # Check if all relations are available
            if relations_in_rule.issubset(available_relations):
                rule_with_type = rule.copy()
                rule_with_type['type'] = 'composition'
                valid_rules.append(rule_with_type)
        
        # Save composition rules
        rules_path = os.path.join(self.processed_dir, "composition_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(valid_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "composition_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in valid_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: composition\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(valid_rules)} composition rules")

    def create_hierarchy_rules(self):
        """Create hierarchy-related rules"""
        print("\nCreating hierarchy rules...")
        
        hierarchy_rules = [
            {
                'name': 'disease_subclass_transitive',
                'type': 'transitive',
                'relation': 'DIRECT_PARENT',
                'domain': 'disease',
                'range': 'disease',
                'confidence': 0.95,
                'description': 'Disease hierarchy is transitive'
            },
            {
                'name': 'pathway_subclass_transitive',
                'type': 'transitive',
                'relation': 'HAS_PARENT_PATHWAY',
                'domain': 'pathway',
                'range': 'pathway',
                'confidence': 0.95,
                'description': 'Pathway hierarchy is transitive'
            },
            {
                'name': 'cell_type_hierarchy',
                'type': 'transitive',
                'relation': 'CELL_TYPE_PARENT',
                'domain': 'cell',
                'range': 'cell',
                'confidence': 0.9,
                'description': 'Cell type hierarchy is transitive'
            },
            {
                'name': 'disease_subclass_inheritance',
                'body': [('disease', 'DIRECT_PARENT', 'disease2')],
                'head': ('disease', 'INHERITS_FROM', 'disease2'),
                'type': 'hierarchical',
                'confidence': 0.85,
                'description': 'Diseases inherit properties from parent diseases'
            }
        ]
        
        # Save hierarchy rules
        rules_path = os.path.join(self.processed_dir, "hierarchy_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(hierarchy_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "hierarchy_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in hierarchy_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Rule: {rule}\n")
                f.write(f"Confidence: {rule.get('confidence', 0.9)}\n")
                f.write(f"Description: {rule.get('description', '')}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(hierarchy_rules)} hierarchy rules")

    def create_domain_specific_rules(self):
        """Create domain-specific biological rules"""
        print("\nCreating domain-specific rules...")
        
        domain_rules = [
            # Drug-related rules
            {
                'name': 'drug_target_interaction',
                'body': [('drug', 'DRUG_TARGET', 'protein')],
                'head': ('protein', 'targeted_by', 'drug'),
                'type': 'domain_specific',
                'confidence': 0.9,
                'domain': 'drug_target',
                'description': 'Drugs target proteins'
            },
            {
                'name': 'drug_metabolism',
                'body': [('drug', 'METABOLIZED_BY', 'protein')],
                'head': ('protein', 'METABOLIZES', 'drug'),
                'type': 'domain_specific',
                'confidence': 0.85,
                'domain': 'drug_metabolism',
                'description': 'Enzymes metabolize drugs'
            },
            
            # Protein-related rules
            {
                'name': 'protein_function_inference',
                'body': [('protein', 'SEQUENCE_SIMILARITY', 'protein2'),
                         ('protein2', 'HAS_FUNCTION', 'function')],
                'head': ('protein', 'HAS_FUNCTION', 'function'),
                'type': 'domain_specific',
                'confidence': 0.7,
                'domain': 'protein_function',
                'description': 'Proteins with similar sequences may share functions'
            },
            {
                'name': 'protein_complex_membership',
                'body': [('protein', 'MEMBER_OF_COMPLEX', 'complex')],
                'head': ('complex', 'HAS_MEMBER', 'protein'),
                'type': 'domain_specific',
                'confidence': 0.95,
                'domain': 'protein_complex',
                'description': 'Complexes have protein members'
            },
            
            # Disease-related rules
            {
                'name': 'disease_symptom_overlap',
                'body': [('disease', 'HAS_SYMPTOM', 'symptom'),
                         ('symptom', 'OBSERVED_IN', 'disease2')],
                'head': ('disease', 'SIMILAR_TO', 'disease2'),
                'type': 'domain_specific',
                'confidence': 0.6,
                'domain': 'disease_similarity',
                'description': 'Diseases sharing symptoms may be similar'
            },
            
            # Pathway-related rules
            {
                'name': 'pathway_crosstalk',
                'body': [('pathway', 'SHARES_PROTEIN', 'pathway2')],
                'head': ('pathway', 'CROSSTALK_WITH', 'pathway2'),
                'type': 'domain_specific',
                'confidence': 0.75,
                'domain': 'pathway_interaction',
                'description': 'Pathways that share proteins may crosstalk'
            }
        ]
        
        # Save domain-specific rules
        rules_path = os.path.join(self.processed_dir, "domain_specific_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(domain_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "domain_specific_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in domain_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Domain: {rule.get('domain', 'general')}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(domain_rules)} domain-specific rules")

    def create_pathway_rules(self):
        """Create pathway-specific rules"""
        print("\nCreating pathway rules...")
        
        pathway_rules = [
            {
                'name': 'pathway_component_inference',
                'body': [('pathway', 'CONTAINS_REACTION', 'reaction'),
                         ('reaction', 'INVOLVES_PROTEIN', 'protein')],
                'head': ('pathway', 'INVOLVES_PROTEIN', 'protein'),
                'type': 'pathway',
                'confidence': 0.9,
                'description': 'Pathways involve proteins through reactions'
            },
            {
                'name': 'pathway_regulation',
                'body': [('pathway', 'REGULATED_BY', 'protein')],
                'head': ('protein', 'REGULATES_PATHWAY', 'pathway'),
                'type': 'pathway',
                'confidence': 0.85,
                'description': 'Proteins regulate pathways'
            },
            {
                'name': 'pathway_hierarchy_inference',
                'body': [('pathway', 'HAS_SUBPATHWAY', 'pathway2'),
                         ('pathway2', 'INVOLVES_PROTEIN', 'protein')],
                'head': ('pathway', 'INVOLVES_PROTEIN', 'protein'),
                'type': 'pathway',
                'confidence': 0.8,
                'description': 'Proteins in subpathways are involved in parent pathways'
            }
        ]
        
        # Save pathway rules
        rules_path = os.path.join(self.processed_dir, "pathway_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(pathway_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "pathway_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in pathway_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(pathway_rules)} pathway rules")

    def create_disease_rules(self):
        """Create disease-specific rules"""
        print("\nCreating disease rules...")
        
        disease_rules = [
            {
                'name': 'disease_genetic_basis',
                'body': [('disease', 'ASSOCIATED_WITH_GENE', 'gene'),
                         ('gene', 'MUTATED_IN', 'disease')],
                'head': ('disease', 'HAS_GENETIC_BASIS', 'gene'),
                'type': 'disease',
                'confidence': 0.9,
                'description': 'Diseases have genetic basis through associated genes'
            },
            {
                'name': 'disease_phenotype_similarity',
                'body': [('disease', 'HAS_PHENOTYPE', 'phenotype'),
                         ('phenotype', 'OBSERVED_IN', 'disease2')],
                'head': ('disease', 'PHENOTYPICALLY_SIMILAR', 'disease2'),
                'type': 'disease',
                'confidence': 0.8,
                'description': 'Diseases sharing phenotypes are phenotypically similar'
            },
            {
                'name': 'disease_comorbidity',
                'body': [('disease', 'SHARES_PATHWAY', 'disease2')],
                'head': ('disease', 'COMORBID_WITH', 'disease2'),
                'type': 'disease',
                'confidence': 0.7,
                'description': 'Diseases sharing pathways may be comorbid'
            }
        ]
        
        # Save disease rules
        rules_path = os.path.join(self.processed_dir, "disease_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(disease_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "disease_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in disease_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(disease_rules)} disease rules")

    def create_drug_rules(self):
        """Create drug-specific rules"""
        print("\nCreating drug rules...")
        
        drug_rules = [
            {
                'name': 'drug_similarity_structure',
                'body': [('drug', 'CHEMICAL_SIMILARITY', 'drug2')],
                'head': ('drug', 'STRUCTURALLY_SIMILAR', 'drug2'),
                'type': 'drug',
                'confidence': 0.9,
                'description': 'Drugs with chemical similarity are structurally similar'
            },
            {
                'name': 'drug_similarity_target',
                'body': [('drug', 'SHARES_TARGET', 'drug2')],
                'head': ('drug', 'TARGET_SIMILAR', 'drug2'),
                'type': 'drug',
                'confidence': 0.85,
                'description': 'Drugs sharing targets are target-similar'
            },
            {
                'name': 'drug_side_effect_similarity',
                'body': [('drug', 'SHARES_SIDE_EFFECT', 'drug2')],
                'head': ('drug', 'SIDE_EFFECT_SIMILAR', 'drug2'),
                'type': 'drug',
                'confidence': 0.8,
                'description': 'Drugs sharing side effects are side-effect-similar'
            },
            {
                'name': 'drug_repurposing_candidate',
                'body': [('drug', 'TARGETS_PROTEIN', 'protein'),
                         ('protein', 'ASSOCIATED_WITH_DISEASE', 'disease')],
                'head': ('drug', 'POTENTIAL_TREATMENT_FOR', 'disease'),
                'type': 'drug',
                'confidence': 0.6,
                'description': 'Drugs may be repurposed for diseases through shared targets'
            }
        ]
        
        # Save drug rules
        rules_path = os.path.join(self.processed_dir, "drug_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(drug_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "drug_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in drug_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(drug_rules)} drug rules")

    def create_protein_rules(self):
        """Create protein-specific rules"""
        print("\nCreating protein rules...")
        
        protein_rules = [
            {
                'name': 'protein_interaction_network',
                'body': [('protein', 'INTERACTS_WITH', 'protein2')],
                'head': ('protein2', 'INTERACTS_WITH', 'protein'),
                'type': 'protein',
                'confidence': 0.95,
                'description': 'Protein interactions are bidirectional'
            },
            {
                'name': 'protein_family_membership',
                'body': [('protein', 'BELONGS_TO_FAMILY', 'family'),
                         ('family', 'HAS_MEMBER', 'protein2')],
                'head': ('protein', 'SIMILAR_TO', 'protein2'),
                'type': 'protein',
                'confidence': 0.85,
                'description': 'Proteins in same family are similar'
            },
            {
                'name': 'protein_functional_association',
                'body': [('protein', 'CO_EXPRESSED_WITH', 'protein2')],
                'head': ('protein', 'FUNCTIONALLY_ASSOCIATED', 'protein2'),
                'type': 'protein',
                'confidence': 0.75,
                'description': 'Co-expressed proteins may be functionally associated'
            },
            {
                'name': 'protein_pathway_involvement',
                'body': [('protein', 'PARTICIPATES_IN', 'pathway')],
                'head': ('pathway', 'HAS_PARTICIPANT', 'protein'),
                'type': 'protein',
                'confidence': 0.9,
                'description': 'Proteins participate in pathways'
            }
        ]
        
        # Save protein rules
        rules_path = os.path.join(self.processed_dir, "protein_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(protein_rules, f)
        
        # Save text version
        txt_path = os.path.join(self.processed_dir, "protein_rules.txt")
        with open(txt_path, 'w') as f:
            for rule in protein_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"  Created {len(protein_rules)} protein rules")

    def create_combined_rules(self):
        """Create a combined rules file with all rules"""
        print("\nCreating combined rules file...")
        
        combined_rules = []
        
        # Collect all rules from individual files
        rule_files = [
            'biological_rules.pkl',
            'symmetric_rules.pkl',
            'inverse_rules.pkl',
            'composition_rules.pkl',
            'hierarchy_rules.pkl',
            'domain_specific_rules.pkl',
            'pathway_rules.pkl',
            'disease_rules.pkl',
            'drug_rules.pkl',
            'protein_rules.pkl'
        ]
        
        for rule_file in rule_files:
            file_path = os.path.join(self.processed_dir, rule_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        rules = pickle.load(f)
                        combined_rules.extend(rules)
                    print(f"  Added {len(rules)} rules from {rule_file}")
                except:
                    print(f"  Warning: Could not load {rule_file}")
        
        # Save combined rules
        combined_path = os.path.join(self.processed_dir, "all_rules.pkl")
        with open(combined_path, 'wb') as f:
            pickle.dump(combined_rules, f)
        
        # Save combined text version
        combined_txt_path = os.path.join(self.processed_dir, "all_rules.txt")
        with open(combined_txt_path, 'w') as f:
            for rule in combined_rules:
                f.write(f"Name: {rule.get('name', 'unnamed')}\n")
                f.write(f"Type: {rule.get('type', 'unknown')}\n")
                f.write(f"Rule: {rule}\n")
                f.write(f"Confidence: {rule.get('confidence', 0.5)}\n")
                f.write(f"Description: {rule.get('description', '')}\n")
                f.write("=" * 60 + "\n")
        
        print(f"\n  Created combined rules file with {len(combined_rules)} total rules")

    def create_biological_rules(self):  # Keep your original method but enhance it
        """Create domain-specific biological rules for reasoning (enhanced version)"""
        print("\n" + "=" * 60)
        print("Creating biological rules...")
        print("=" * 60)
        
        # Your existing rules (I'll keep them but add more)
        rules = [
            # Your existing rules here...
            {
                'name': 'drug_indication_inverse',
                'body': [('drug', 'DRUG_INDICATION_ASSOCIATION', 'disease')],
                'head': ('disease', 'inverse_DRUG_INDICATION_ASSOCIATION', 'drug'),
                'type': 'inverse',
                'confidence': 0.9
            },
            # ... (keep all your existing rules)
        ]
        
        # Add more biological rules
        additional_rules = [
            {
                'name': 'protein_domain_interaction',
                'body': [('protein', 'HAS_DOMAIN', 'domain'),
                         ('domain', 'INTERACTS_WITH', 'domain2'),
                         ('domain2', 'IN_PROTEIN', 'protein2')],
                'head': ('protein', 'PPI', 'protein2'),
                'type': 'biological',
                'confidence': 0.7,
                'description': 'Proteins interact through their domains'
            },
            {
                'name': 'drug_metabolism_pathway',
                'body': [('drug', 'METABOLIZED_BY', 'enzyme'),
                         ('enzyme', 'IN_PATHWAY', 'pathway')],
                'head': ('drug', 'AFFECTS_PATHWAY', 'pathway'),
                'type': 'biological',
                'confidence': 0.75,
                'description': 'Drugs affect pathways through metabolizing enzymes'
            },
            {
                'name': 'disease_pathway_disruption',
                'body': [('disease', 'ASSOCIATED_WITH_GENE', 'gene'),
                         ('gene', 'PRODUCT', 'protein'),
                         ('protein', 'IN_PATHWAY', 'pathway')],
                'head': ('disease', 'DISRUPTS_PATHWAY', 'pathway'),
                'type': 'biological',
                'confidence': 0.8,
                'description': 'Diseases disrupt pathways through associated genes and proteins'
            }
        ]
        
        rules.extend(additional_rules)
        
        # Filter rules based on available relations
        available_relations = set(self.relation2id.keys())
        valid_rules = []
        
        for rule in rules:
            # Collect all relations in this rule
            relations_in_rule = set()
            
            # Add body relations
            for body_elem in rule['body']:
                if len(body_elem) >= 2:
                    relations_in_rule.add(body_elem[1])
            
            # Add head relation
            if len(rule['head']) >= 2:
                relations_in_rule.add(rule['head'][1])
            
            # Check if all relations are available
            if relations_in_rule.issubset(available_relations):
                valid_rules.append(rule)
            else:
                missing = relations_in_rule - available_relations
                print(f"  Skipping rule '{rule['name']}': missing relations {missing}")
        
        # Save rules to file
        rules_path = os.path.join(self.processed_dir, "biological_rules.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(valid_rules, f)
        print(f"  Saved {len(valid_rules)} rules to {rules_path}")
        
        # Also save a text version for inspection
        rules_txt_path = os.path.join(self.processed_dir, "biological_rules.txt")
        with open(rules_txt_path, 'w') as f:
            for rule in valid_rules:
                f.write(f"Name: {rule['name']}\n")
                f.write(f"Type: {rule['type']}\n")
                f.write(f"Body: {rule['body']}\n")
                f.write(f"Head: {rule['head']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                f.write(f"Description: {rule.get('description', '')}\n")
                f.write("-" * 40 + "\n")
        
        return valid_rules


    def process(self):
        """Run the complete processing pipeline"""
        print("BioKG Complete Data Processor")
        print("=" * 60)
        
        # Step 1: Load links first to get base entities and relations
        self.load_links()
        
        # Step 2: Load additional entities from property files
        self.load_all_entities_from_properties()
        
        # Step 3: Load additional entities from metadata files
        self.load_all_entities_from_metadata()
        
        # Step 4: Create ID mappings
        self.create_mappings()
        
        # Step 5: Group entities by type
        self.create_entity_type_groups()
        
        # Step 6: Create type to IDs mapping
        self.create_type_to_ids_mapping()
        
        # Step 7: Create train/valid/test splits
        self.create_train_valid_test_splits()
        
        # Step 8: Save text files (with human-readable names)
        self.save_text_files()
        
        # Step 9: Save pickle files
        self.save_pickle_files()
        
        # Step 10: Save numpy files
        self.save_numpy_files()
        
        # Step 11: Create all rule files (replaces just create_biological_rules)
        self.create_all_rule_files()
        
        print("\n" + "=" * 60)
        print("✅ BioKG data processing complete!")
        print("=" * 60)
        print("\nGenerated files:")
        print(f"  Text files in {self.data_dir}/:")
        print("    - entities.txt (with human-readable names)")
        print("    - relations.txt")
        print("    - train.txt (with human-readable names)")
        print("    - valid.txt (with human-readable names)")
        print("    - test.txt (with human-readable names)")
        print(f"\n  Pickle files in {self.processed_dir}/:")
        print("    - entity2id.pkl, id2entity.pkl")
        print("    - entity2name.pkl (human-readable names)")
        print("    - relation2id.pkl, id2relation.pkl")
        print("    - entities_by_type.pkl")
        print("    - type_to_ids.pkl, id_to_type.pkl")
        print("    - entity_properties.pkl")
        print("    - entity_metadata.pkl")
        print("    - stats.pkl")
        print("    - biological_rules.pkl")
        print("    - symmetric_rules.pkl")
        print("    - inverse_rules.pkl")
        print("    - composition_rules.pkl")
        print("    - hierarchy_rules.pkl")
        print("    - domain_specific_rules.pkl")
        print("    - pathway_rules.pkl")
        print("    - disease_rules.pkl")
        print("    - drug_rules.pkl")
        print("    - protein_rules.pkl")
        print("    - all_rules.pkl (combined)")
        print(f"\n  Numpy files in {self.processed_dir}/:")
        print("    - train_triples.npy")
        print("    - valid_triples.npy")
        print("    - test_triples.npy")


def main():
    # Set directories
    data_dir = "raw"
    processed_dir = "processed"
    
    # Create processor and run
    processor = BioKGDataProcessor(data_dir, processed_dir)
    processor.process()


if __name__ == "__main__":
    main()
