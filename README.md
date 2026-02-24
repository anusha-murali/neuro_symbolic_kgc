# NeuroSymbolic KGC

> A neuro-symbolic framework for knowledge graph completion on biological data (BioKG), combining complex embeddings with mined biological rules

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**NeuroSymbolic KGC** is a comprehensive framework for knowledge graph completion on biological data, specifically designed for the BioKG dataset. It integrates neural embeddings with symbolic biological rules to achieve state-of-the-art performance on link prediction tasks.

### Key Features

- **Neural Component**: ComplEx embeddings in complex space for learning latent patterns in biological relationships
- **Symbolic Component**: Integration of mined biological rules (inverse, symmetric, chain, composition) with learnable weights
- **Rule Mining**: Automatic extraction of biological rules directly from the graph structure using co-occurrence patterns
- **Filtered Evaluation**: Standard KGC evaluation protocol with masking of known true triples
- **Self-Adversarial Sampling**: Hard negative generation for more effective training
- **Mixed Precision**: FP16 training support for A100 GPUs
- **Type-Constrained Sampling**: Entity type-based negative sampling for biologically plausible negatives

### Performance

After optimization, the model achieves:
- **MRR**: 0.163 (filtered)
- **Hits@1**: 9.8%
- **Hits@3**: 17.3%
- **Hits@10**: 29.3%

---

## Quick Start

### Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/anusha-murali/neuro_symbolic_kgc.git
cd neuro_symbolic_kgc

# Create and activate conda environment
conda create -n neuro_symbolic python=3.10 -y
conda activate neuro_symbolic

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Quick Start with Sample Data

```bash
# Download and prepare BioKG data (if not already present)
# Place data files in data/processed/ directory:
# - train_triples.npy, valid_triples.npy, test_triples.npy
# - entity2id.pkl, relation2id.pkl
# - id_to_type.pkl, type_to_ids.pkl (optional, for type-constrained sampling)

# Run training with default configuration
python src/main.py --config config.yaml

# Train with custom parameters
python src/main.py \
  --config config.yaml \
  --model.embedding_dim=256 \
  --training.batch_size=1024 \
  --training.learning_rate=0.001

# Evaluate a trained model
python src/main.py --mode eval --config config.yaml

# Fast development mode (small subset)
python src/main.py --fast_dev --config config.yaml
```

### Google Colab Setup

For running on Google Colab with A100 GPU:

```python
# In Colab cell
!git clone https://github.com/anusha-murali/neuro_symbolic_kgc.git
%cd neuro_symbolic_kgc
!pip install -r requirements.txt
!python src/main.py --config config.yaml
```

---

## Project Structure

```
neuro_symbolic_kgc/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── config.yaml                         # Main configuration file
│
├── src/                                # Source code
│   ├── main.py                         # Main training/evaluation script
│   │
│   ├── models/                          # Model implementations
│   │   └── neuro_symbolic.py            # NeuroSymbolicKGC and ComplEx
│   │
│   ├── utils/                           # Utility modules
│   │   ├── data_loader.py               # BioKGDataset and FastNegativeSampler
│   │   ├── rule_miner.py                 # BiologicalRuleMiner for rule extraction
│   │   └── relation_mapper.py            # Relation name mapping utilities
│   │
│   └── __init__.py                       # Package initialization
│
├── scripts/                             # Executable scripts
│   └── run_experiments.sh                # Batch experiment runner
│
├── notebooks/                            # Analysis notebooks
│   ├── 01_data_exploration.ipynb         # Explore BioKG dataset
│   ├── 02_rule_analysis.ipynb             # Analyze mined rules
│   └── 03_results_visualization.ipynb     # Visualize results
│
├── tests/                                # Unit tests
│   ├── test_model.py                      # Model tests
│   ├── test_data_loader.py                 # Data loader tests
│   └── test_rule_miner.py                  # Rule miner tests
│
├── data/                                  # Data directory (gitignored)
│   └── processed/                          # Processed BioKG data
│       ├── train_triples.npy               # Training triples
│       ├── valid_triples.npy               # Validation triples
│       ├── test_triples.npy                # Test triples
│       ├── entity2id.pkl                    # Entity ID mappings
│       ├── relation2id.pkl                  # Relation ID mappings
│       ├── id_to_type.pkl                   # Entity type mappings
│       ├── type_to_ids.pkl                  # Type to entity IDs
│       └── biological_rules.pkl             # Pre-mined rules (optional)
│
├── checkpoints/                           # Model checkpoints (gitignored)
│   ├── best_model.pt                       # Best model based on validation MRR
│   └── latest.pt                           # Latest model checkpoint
│
├── logs/                                   # Training logs (gitignored)
│   └── run_YYYYMMDD_HHMMSS/                 # Timestamped run directories
│       ├── config.yaml                       # Copy of config
│       ├── metrics.json                       # Training metrics
│       └── stdout.log                          # Console output
│
└── docs/                                   # Documentation
    ├── installation.md                       # Detailed installation guide
    ├── dataset_preparation.md                 # BioKG dataset setup
    └── evaluation.md                           # Evaluation protocols
```

---

## Models

### NeuroSymbolicKGC

**Purpose**: Combined neural-symbolic model for knowledge graph completion

**Key Features**:
- ComplEx neural component with complex embeddings
- Symbolic rule integration with learnable weights
- Self-adversarial negative sampling loss
- N3 regularization for embedding stability

**Architecture**:

```python
# Neural score (ComplEx)
neural_score = Re(⟨h, r, conj(t)⟩)

# Symbolic score
symbolic_score = tanh(rule_confidence[r] * rule_weight)

# Combined score
final_score = neural_score + lambda_logic * symbolic_score
```

**Configuration**: See `config.yaml`

### ComplEx

**Purpose**: Complex embedding model for knowledge graph completion

**Key Features**:
- Entities and relations as complex vectors
- Captures symmetric and antisymmetric relations
- Efficient batched prediction for all entities

**Mathematical Formulation**:
```
score(h,r,t) = Re( (h_re + i*h_im) * (r_re + i*r_im) · conj(t_re + i*t_im) )
             = (h_re·r_re - h_im·r_im)·t_re + (h_re·r_im + h_im·r_re)·t_im
```

---

## BioKG Dataset

### Overview

BioKG is a comprehensive biological knowledge graph that integrates curated relational data from multiple open-source databases including UniProt, DrugBank, KEGG, and Reactome. It focuses specifically on biological relationships while preserving essential entity information.

### Dataset Statistics

| Split | Triples | Entities | Relations |
|-------|---------|----------|-----------|
| **Train** | 1,654,398 | 345,690 | 40 |
| **Validation** | 206,799 | 345,690 | 40 |
| **Test** | 206,801 | 345,690 | 40 |

### Entity Types

- Genes and proteins
- Drugs and chemical compounds
- Diseases and genetic disorders
- Pathways and biological processes
- Tissues and cell lines
- GO terms (Biological Process, Cellular Component, Molecular Function)

### Relation Types

Sample of 40 relations including:
- `PPI` (Protein-Protein Interaction)
- `DPI` (Drug-Protein Interaction)
- `DDI` (Drug-Drug Interaction)
- `DRUG_TARGET`
- `PROTEIN_PATHWAY_ASSOCIATION`
- `PROTEIN_DISEASE_ASSOCIATION`
- `GO_BP`, `GO_CC`, `GO_MF`
- `MEMBER_OF_COMPLEX`
- `PROTEIN_EXPRESSED_IN`

### Data Format

The dataset is provided as:
- **Triples**: Numpy arrays of shape (n_triples, 3) with integer IDs
- **Mappings**: Pickle files mapping string IDs to integers
- **Type information**: Entity type mappings for type-constrained sampling

### Data Preparation

```bash
# Download BioKG data (if not already available)
# Place files in data/processed/ with the following structure:
data/processed/
├── train_triples.npy
├── valid_triples.npy
├── test_triples.npy
├── entity2id.pkl
├── relation2id.pkl
├── id_to_type.pkl
└── type_to_ids.pkl
```

---

## Dependencies

### Core Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.10+ | Runtime |
| **PyTorch** | 2.0+ | Deep learning framework |
| **NumPy** | 1.24+ | Numerical operations |
| **tqdm** | 4.65+ | Progress bars |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **wandb** | Latest | Experiment tracking (optional) |
| **jupyter** | Latest | Analysis notebooks |
| **matplotlib** | Latest | Visualization |
| **seaborn** | Latest | Statistical visualizations |

### Installation

```bash
# Install core dependencies
pip install torch numpy tqdm pyyaml

# Install all dependencies (including optional)
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
wandb>=0.15.0  # optional
jupyter>=1.0.0  # optional
matplotlib>=3.7.0  # optional
seaborn>=0.12.0  # optional
```

### GPU Requirements

| GPU | Memory | Batch Size | Performance |
|-----|--------|------------|-------------|
| **A100 40GB** | 40GB | 1024-2048 | Full training |
| **V100 32GB** | 32GB | 512-1024 | Reduced batch size |
| **T4 16GB** | 16GB | 256-512 | Development only |
| **CPU** | - | 64-128 | Debugging only |

### CUDA Compatibility

- **CUDA 11.8+** required for PyTorch 2.0+
- **cuDNN 8.7+** recommended for optimal performance
- Tested on A100 (80GB) with CUDA 12.8

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{neuro_symbolic_kgc,
  title={NeuroSymbolic KGC: A Framework for Knowledge Graph Completion on Biological Data},
  author={Anusha Murali},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/neuro_symbolic_kgc}
}
```

## Acknowledgments

- BioKG dataset from [Walsh et al. (CIKM 2020)](https://doi.org/10.1145/3340531.3412776)
- ComplEx implementation inspired by [Trouillon et al. (ICML 2016)](https://proceedings.mlr.press/v48/trouillon16.html)
