"""
HLA Sequence Encoders
---------------------
This module provides different encoder implementations for HLA protein sequences.

Available encoders:
- ProtBERTEncoder: BERT-based model from RostLab (420M params)
- ESMEncoder: ESM-2 model from Meta AI (650M params)
- ProtT5Encoder: T5-based model from RostLab (1.3B params)
- AnkhEncoder: Purpose-built protein model (50M base, 650M large)
"""
from .protbert import ProtBERTEncoder
from .esm import ESMEncoder
from .prott5 import ProtT5Encoder
from .ankh import AnkhEncoder

__all__ = [
    "ProtBERTEncoder",
    "ESMEncoder",
    "ProtT5Encoder",
    "AnkhEncoder",
]
