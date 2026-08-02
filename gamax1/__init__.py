"""
GamaX1 -- first working version of the Aetherion architecture as a
real, trainable NLP language model.

See the Aetherion Technical Report (v4) for the full research program
this implementation is built on; see README.md in this package for
how each research finding maps to code here.
"""

from .model import GamaX1Model, GamaX1Block, AetherionFFN, CausalSelfAttention
from .layers import (
    SparseSuperpositionLinear,
    DynamicSparsityController,
    ProbationaryMemoryTracker,
    HexNeighborInfluence,
    RouterExpert,
    ValidatorExpert,
    build_hex_neighbor_table,
)
from .tokenizer import BPETokenizer, CharTokenizer

__version__ = "1.0.0"
__all__ = [
    "GamaX1Model", "GamaX1Block", "AetherionFFN", "CausalSelfAttention",
    "SparseSuperpositionLinear", "DynamicSparsityController",
    "ProbationaryMemoryTracker", "HexNeighborInfluence",
    "RouterExpert", "ValidatorExpert", "build_hex_neighbor_table",
    "CharTokenizer", "BPETokenizer",
]
