"""
NeuroSymbolic KGC: A neuro-symbolic framework for knowledge graph completion on biological data.

This package provides tools for training and evaluating knowledge graph completion models
on the BioKG dataset, combining neural embeddings (ComplEx) with symbolic biological rules.
"""

__version__ = "1.0.0"
__author__ = "Anusha Murali"
__email__ = "anusha.murali.gr@dartmouth.edu"
__license__ = "MIT"

# Import main classes for easy access
from src.models.neuro_symbolic import NeuroSymbolicKGC, ComplEx
from src.utils.data_loader import BioKGDataset, FastNegativeSampler, create_dataloader
from src.utils.rule_miner import BiologicalRuleMiner
from src.utils.relation_mapper import RelationMapper, get_direct_mapping

__all__ = [
    # Models
    'NeuroSymbolicKGC',
    'ComplEx',
    
    # Data loading
    'BioKGDataset',
    'FastNegativeSampler',
    'create_dataloader',
    
    # Rule mining
    'BiologicalRuleMiner',
    
    # Relation mapping
    'RelationMapper',
    'get_direct_mapping',
]
