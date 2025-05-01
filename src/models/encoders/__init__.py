"""
    HLA Sequence Encoders
    ---------------------
    This module provides different encoder implementations for HLA protein sequences.
"""
from .protbert import ProtBERTEncoder
from .esm import ESMEncoder # Updated import

__all__ = [
    "ProtBERTEncoder",
    "ESMEncoder" # Updated class name
]
