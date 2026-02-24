"""
Model implementations for NeuroSymbolic KGC.

This module contains the neural and neuro-symbolic models for knowledge graph completion.
"""

from .neuro_symbolic import NeuroSymbolicKGC, ComplEx

__all__ = [
    'NeuroSymbolicKGC',
    'ComplEx',
]
