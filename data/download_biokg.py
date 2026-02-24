import os
import subprocess
import sys
import pickle
import numpy as np
from tqdm import tqdm

def check_install_zenodo_get():
    """Check if zenodo_get is installed, if not, install it"""
    try:
        subprocess.run(['zenodo_get', '--version'], capture_output=True, check=True)
        print("✓ zenodo_get is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing zenodo_get...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'zenodo-get'], check=True)
            print("✓ zenodo_get installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install zenodo_get: {e}")
            return False

def download_biokg_zenodo():
    """Download BioKG dataset using zenodo_get"""
    print("=" * 60)
    print("Downloading BioKG from Zenodo (Record 8005711)")
    print("=" * 60)
    
    # Check and install zenodo_get if needed
    if not check_install_zenodo_get():
        return False
    
    # Create data directory
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # Zenodo record ID for BioKG
    zenodo_record = "8005711"
    
    # Change to data directory and download
    original_dir = os.getcwd()
    os.chdir(data_dir)
    
    try:
        print(f"\nDownloading to {data_dir}/...")
        
        # Run zenodo_get command
        cmd = ['zenodo_get', zenodo_record]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Download failed with error:")
            print(result.stderr)
            return False
        
        print("✓ Download completed successfully")
        
        # List downloaded files
        downloaded_files = os.listdir('.')
        print(f"\nDownloaded files:")
        for file in sorted(downloaded_files):
            if os.path.isfile(file):
                size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
                print(f"  - {file} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False
    finally:
        os.chdir(original_dir)

def extract_files(data_dir="data/raw"):
    """Extract any zip files if present"""
    print("\nChecking for zip files to extract...")
    
    for file in os.listdir(data_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(data_dir, file)
            print(f"Extracting {file}...")
            
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print(f"✓ Extracted {file}")
            
            # Optionally remove the zip file after extraction
            # os.remove(zip_path)

def verify_downloaded_files(data_dir="data/raw"):
    """Verify that all required files are present"""
    required_files = [
        "entities.tsv", "relations.tsv", 
        "train.tsv", "valid.tsv", "test.tsv"
    ]
    
    # Alternative filenames that might be used
    alternative_names = {
        "entities.tsv": ["entities.txt", "entity.txt", "nodes.txt"],
        "relations.tsv": ["relations.txt", "relation.txt", "edges.txt"],
        "train.tsv": ["train.txt", "training.txt"],
        "valid.tsv": ["valid.txt", "validation.txt", "dev.txt"],
        "test.tsv": ["test.txt", "testing.txt"]
    }
    
    found_files = {}
    
    for required in required_files:
        if os.path.exists(os.path.join(data_dir, required)):
            found_files[required] = required
        else:
            # Check for alternatives
            for alt in alternative_names.get(required, []):
                if os.path.exists(os.path.join(data_dir, alt)):
                    found_files[required] = alt
                    print(f"Found alternative: {alt} for {required}")
                    break
        
        if required not in found_files:
            print(f"✗ Missing required file: {required}")
            return False
    
    print(f"✓ All required files found: {list(found_files.values())}")
    return found_files

def preprocess_biokg(file_mapping, data_dir="data/raw", processed_dir="data/processed"):
    """Preprocess BioKG data into the format expected by the model"""
    print("\n" + "=" * 60)
    print("Preprocessing BioKG data")
    print("=" * 60)
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Determine the actual filenames
    entities_file = os.path.join(data_dir, file_mapping.get("entities.tsv", "entities.tsv"))
    relations_file = os.path.join(data_dir, file_mapping.get("relations.tsv", "relations.tsv"))
    train_file = os.path.join(data_dir, file_mapping.get("train.tsv", "train.tsv"))
    valid_file = os.path.join(data_dir, file_mapping.get("valid.tsv", "valid.tsv"))
    test_file = os.path.join(data_dir, file_mapping.get("test.tsv", "test.tsv"))
    
    # Load entities and relations
    print("\nLoading entities...")
    with open(entities_file, 'r') as f:
        # Handle different possible formats (with or without headers)
        first_line = f.readline().strip()
        f.seek(0)
        
        if first_line.startswith('entity') or first_line.startswith('id'):
            # Skip header
            entities = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                       for line in f.readlines()[1:]]
        else:
            entities = [line.strip() for line in f]
    
    print(f"Loading relations...")
    with open(relations_file, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        if first_line.startswith('relation') or first_line.startswith('id'):
            # Skip header
            relations = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                        for line in f.readlines()[1:]]
        else:
            relations = [line.strip() for line in f]
    
    print(f"Loaded {len(entities)} entities and {len(relations)} relations")
    
    # Create mappings
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    
    # Save mappings
    with open(os.path.join(processed_dir, "entity2id.pkl"), 'wb') as f:
        pickle.dump(entity2id, f)
    with open(os.path.join(processed_dir, "relation2id.pkl"), 'wb') as f:
        pickle.dump(relation2id, f)
    
    # Also save as text files for easy inspection
    with open(os.path.join(processed_dir, "entities.txt"), 'w') as f:
        f.write('\n'.join(entities))
    with open(os.path.join(processed_dir, "relations.txt"), 'w') as f:
        f.write('\n'.join(relations))
    
    # Process triples for each split
    split_stats = {}
    
    for split_name, split_file in [('train', train_file), ('valid', valid_file), ('test', test_file)]:
        print(f"\nProcessing {split_name} split...")
        triples = []
        skipped = 0
        
        with open(split_file, 'r') as f:
            # Check for header
            first_line = f.readline().strip()
            f.seek(0)
            
            lines = f.readlines()
            if first_line.startswith('head') or first_line.startswith('subject'):
                # Skip header
                lines = lines[1:]
            
            for line in tqdm(lines, desc=f"  Processing {split_name}"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[:3]
                        
                        if h in entity2id and r in relation2id and t in entity2id:
                            triples.append([
                                entity2id[h],
                                relation2id[r],
                                entity2id[t]
                            ])
                        else:
                            skipped += 1
                except Exception as e:
                    skipped += 1
                    continue
        
        # Save processed triples
        triples_array = np.array(triples, dtype=np.int32)
        np.save(os.path.join(processed_dir, f"{split_name}_triples.npy"), triples_array)
        
        split_stats[split_name] = {
            'triples': len(triples),
            'skipped': skipped
        }
        
        print(f"  {split_name}: {len(triples)} triples (skipped {skipped})")
    
    # Save dataset statistics
    stats = {
        'n_entities': len(entities),
        'n_relations': len(relations),
        'n_train': split_stats['train']['triples'],
        'n_valid': split_stats['valid']['triples'],
        'n_test': split_stats['test']['triples']
    }
    
    with open(os.path.join(processed_dir, "stats.pkl"), 'wb') as f:
        pickle.dump(stats, f)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    return stats

def main():
    print("BioKG Downloader (Zenodo Version)")
    print("=" * 60)
    
##    # Step 1: Download using zenodo_get
##    if not download_biokg_zenodo():
##        print("\n✗ Failed to download BioKG. Please try manually:")
##        print("  pip install zenodo-get")
##        print("  cd data/raw")
##        print("  zenodo_get 8005711")
##        return
##    
##    # Step 2: Extract any zip files
##    extract_files()
    
##    # Step 3: Verify downloaded files
    data_dir = "data/raw"
    file_mapping = verify_downloaded_files(data_dir)
    
##    if not file_mapping:
##        print("\n✗ Required files not found. Please check the download.")
##        return
    
    # Step 4: Preprocess the data
    stats = preprocess_biokg(file_mapping, data_dir)
    
    print("\n" + "=" * 60)
    print("✅ BioKG download and preprocessing complete!")
    print("=" * 60)
    
    print("\nFiles saved:")
    print(f"  - Raw data: {data_dir}/")
    print(f"  - Processed data: data/processed/")
    print(f"  - Entity mappings: data/processed/entity2id.pkl")
    print(f"  - Relation mappings: data/processed/relation2id.pkl")
    print(f"  - Training triples: data/processed/train_triples.npy")
    print(f"  - Validation triples: data/processed/valid_triples.npy")
    print(f"  - Test triples: data/processed/test_triples.npy")
    
    print("\nNext steps:")
    print("  1. Train the model: ./scripts/train.sh")
    print("  2. Evaluate: ./scripts/evaluate.sh")

if __name__ == "__main__":
    main()
