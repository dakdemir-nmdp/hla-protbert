"""
HLA Sequence Utilities
---------------------
Utilities for preprocessing and manipulating HLA protein sequences.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class SequenceProcessor:
    """Tools for processing and standardizing HLA protein sequences"""
    
    # Standard amino acid alphabet
    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    
    # Special tokens for sequence padding, masking, etc.
    PADDING_CHAR = "X"  # Used for padding shorter sequences
    MASKING_CHAR = "?"  # Used for masking specific positions
    MISSING_CHAR = "-"  # Used for missing/gap positions
    
    def __init__(self, padding_length=None):
        """Initialize sequence processor
        
        Args:
            padding_length: Length to pad sequences to (or None for no padding)
        """
        self.padding_length = padding_length
    
    def standardize_sequence(self, sequence: str) -> str:
        """Standardize a protein sequence
        
        Removes non-standard characters, converts to uppercase, 
        replaces invalid amino acids with standard tokens.
        
        Args:
            sequence: Input protein sequence
            
        Returns:
            Standardized sequence
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Replace non-standard characters
        result = []
        for aa in sequence:
            if aa in self.AA_ALPHABET:
                result.append(aa)
            elif aa in ['.', '*', ' ']:
                # Common indicators of sequence end or gaps
                continue
            elif aa in ['-', '~']:
                # Common gap indicators
                result.append(self.MISSING_CHAR)
            else:
                # Other non-standard characters
                result.append(self.MISSING_CHAR)
                
        return ''.join(result)
    
    def pad_sequence(self, sequence: str, length: Optional[int] = None) -> str:
        """Pad a sequence to specified length
        
        Args:
            sequence: Input sequence
            length: Target length (uses self.padding_length if None)
            
        Returns:
            Padded sequence
        """
        if length is None:
            length = self.padding_length
            
        if length is None or len(sequence) >= length:
            return sequence
            
        # Pad to target length
        padding = self.PADDING_CHAR * (length - len(sequence))
        return sequence + padding
    
    def align_sequences(self, sequences: Dict[str, str]) -> Dict[str, str]:
        """Perform simple sequence alignment for a set of HLA sequences
        
        Note: This is a basic alignment approach. For more sophisticated alignment,
        external tools like MUSCLE or ClustalW should be used.
        
        Args:
            sequences: Dict mapping allele names to sequences
            
        Returns:
            Dict of aligned sequences
        """
        # Find maximum sequence length
        max_length = max(len(seq) for seq in sequences.values())
        
        # Pad all sequences to maximum length
        aligned = {}
        for allele, seq in sequences.items():
            aligned[allele] = self.pad_sequence(seq, max_length)
            
        return aligned
    
    def extract_peptide_binding_region(self, sequence: str, locus: str) -> str:
        """Extract the peptide binding region (PBR) for a given HLA sequence
        
        Args:
            sequence: Complete protein sequence
            locus: HLA locus (A, B, C, DRB1, etc.)
            
        Returns:
            Extracted peptide binding region sequence
        """
        # PBR positions vary by locus - these are approximate and simplified
        if locus in ['A', 'B', 'C']:  # Class I
            # Simplified approach: use residues 1-180 as PBR for class I
            return sequence[:180] if len(sequence) >= 180 else sequence
        elif locus.startswith('DR'):  # Class II DR
            # Simplified approach: use residues 1-90 as PBR for DRB
            return sequence[:90] if len(sequence) >= 90 else sequence
        elif locus.startswith('DQ') or locus.startswith('DP'):  # Class II DQ/DP
            # Simplified approach: use residues 1-90 as PBR for DQ/DP
            return sequence[:90] if len(sequence) >= 90 else sequence
        else:
            # Unknown locus - return full sequence
            logger.warning(f"Unknown locus {locus} for PBR extraction, using full sequence")
            return sequence
    
    def get_amino_acid_features(self, sequence: str) -> np.ndarray:
        """Convert sequence to amino acid property features
        
        Uses a simplified set of biochemical properties for each amino acid:
        - Hydrophobicity
        - Size
        - Charge
        - Polarity
        - Aromaticity
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Array of shape (len(sequence), 5) with amino acid properties
        """
        # Define amino acid properties (simplified)
        # Order: hydrophobicity, size, charge, polarity, aromaticity
        aa_properties = {
            'A': [0.7, -0.5, 0, 0, 0],  # Alanine
            'C': [0.8, -0.3, 0, 0, 0],  # Cysteine
            'D': [-0.7, -0.1, -1, 1, 0],  # Aspartic acid
            'E': [-0.6, 0.1, -1, 1, 0],  # Glutamic acid
            'F': [0.9, 0.7, 0, 0, 1],  # Phenylalanine
            'G': [0.5, -1.0, 0, 0, 0],  # Glycine
            'H': [-0.1, 0.2, 0.5, 1, 1],  # Histidine
            'I': [0.9, 0.4, 0, 0, 0],  # Isoleucine
            'K': [-0.5, 0.3, 1, 1, 0],  # Lysine
            'L': [0.9, 0.4, 0, 0, 0],  # Leucine
            'M': [0.7, 0.3, 0, 0, 0],  # Methionine
            'N': [-0.4, 0.0, 0, 1, 0],  # Asparagine
            'P': [0.0, -0.2, 0, 0, 0],  # Proline
            'Q': [-0.4, 0.1, 0, 1, 0],  # Glutamine
            'R': [-0.6, 0.5, 1, 1, 0],  # Arginine
            'S': [-0.2, -0.4, 0, 1, 0],  # Serine
            'T': [-0.1, -0.2, 0, 1, 0],  # Threonine
            'V': [0.8, 0.0, 0, 0, 0],  # Valine
            'W': [0.7, 0.8, 0, 0, 1],  # Tryptophan
            'Y': [0.4, 0.5, 0, 1, 1],  # Tyrosine
            # Special tokens
            'X': [0, 0, 0, 0, 0],  # Padding
            '?': [0, 0, 0, 0, 0],  # Masking
            '-': [0, 0, 0, 0, 0],  # Missing
        }
        
        # Convert sequence to features
        features = []
        for aa in sequence:
            if aa in aa_properties:
                features.append(aa_properties[aa])
            else:
                # Unknown amino acid - use zeros
                features.append([0, 0, 0, 0, 0])
                
        return np.array(features)
    
    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert sequence to one-hot encoded matrix
        
        Args:
            sequence: Protein sequence
            
        Returns:
            One-hot encoded matrix of shape (len(sequence), len(AA_ALPHABET))
        """
        # Create mapping from amino acid to index
        aa_to_idx = {aa: i for i, aa in enumerate(self.AA_ALPHABET)}
        
        # Create one-hot matrix
        one_hot = np.zeros((len(sequence), len(self.AA_ALPHABET)))
        
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                one_hot[i, aa_to_idx[aa]] = 1
                
        return one_hot

class HLASequenceUtils:
    """Utility functions for working with HLA sequences and alleles"""
    
    @staticmethod
    def parse_allele_name(allele: str) -> Tuple[str, str, str]:
        """Parse HLA allele name into components
        
        Args:
            allele: HLA allele name (e.g., 'A*01:01', 'DRB1*15:01')
            
        Returns:
            Tuple of (locus, group, protein)
            Example: 'A*01:01' -> ('A', '01', '01')
        """
        # Remove 'HLA-' prefix if present
        if allele.startswith('HLA-'):
            allele = allele[4:]
            
        # Split locus and fields
        if '*' in allele:
            locus, fields = allele.split('*', 1)
        else:
            # Handle cases without '*'
            match = re.match(r'([A-Z0-9]+)(\d+)(?::(\d+))?', allele)
            if match:
                locus, group, protein = match.groups()
                if protein is None:
                    protein = ''
                return locus, group, protein
            else:
                locus = allele
                fields = ''
                
        # Extract fields
        field_parts = fields.split(':')
        group = field_parts[0] if len(field_parts) > 0 else ''
        protein = field_parts[1] if len(field_parts) > 1 else ''
        
        return locus, group, protein
    
    @staticmethod
    def is_same_serotype(allele1: str, allele2: str) -> bool:
        """Check if two alleles belong to the same serotype
        
        Args:
            allele1: First HLA allele
            allele2: Second HLA allele
            
        Returns:
            True if alleles belong to same serotype (first field match)
        """
        locus1, group1, _ = HLASequenceUtils.parse_allele_name(allele1)
        locus2, group2, _ = HLASequenceUtils.parse_allele_name(allele2)
        
        return locus1 == locus2 and group1 == group2
    
    @staticmethod
    def is_same_protein(allele1: str, allele2: str) -> bool:
        """Check if two alleles encode the same protein sequence
        
        Args:
            allele1: First HLA allele
            allele2: Second HLA allele
            
        Returns:
            True if alleles encode same protein (first two fields match)
        """
        locus1, group1, protein1 = HLASequenceUtils.parse_allele_name(allele1)
        locus2, group2, protein2 = HLASequenceUtils.parse_allele_name(allele2)
        
        return locus1 == locus2 and group1 == group2 and protein1 == protein2
    
    @staticmethod
    def get_allele_family(allele: str) -> str:
        """Get allele family (first field)
        
        Args:
            allele: HLA allele name
            
        Returns:
            Allele family (e.g., 'A*01' from 'A*01:01')
        """
        locus, group, _ = HLASequenceUtils.parse_allele_name(allele)
        return f"{locus}*{group}" if group else locus
    
    @staticmethod
    def standardize_allele_name(allele: str) -> str:
        """Standardize HLA allele name format
        
        Args:
            allele: HLA allele name in various formats
            
        Returns:
            Standardized allele name (e.g., 'A*01:01')
        """
        # Remove 'HLA-' prefix if present
        if allele.startswith('HLA-'):
            allele = allele[4:]
            
        # Handle different separators and formats
        if '*' not in allele:
            # Convert formats like A0101 to A*01:01
            match = re.match(r'([A-Z0-9]+)(\d{2,4})(?:$|:)', allele)
            if match:
                locus, digits = match.groups()
                if len(digits) == 4:  # A0101 format
                    return f"{locus}*{digits[:2]}:{digits[2:]}"
                elif len(digits) == 2:  # A01 format
                    return f"{locus}*{digits}"
        
        # Handle formats with '*' but no ':'
        if '*' in allele and ':' not in allele:
            parts = allele.split('*')
            locus = parts[0]
            digits = parts[1]
            if len(digits) == 4:  # A*0101 format
                return f"{locus}*{digits[:2]}:{digits[2:]}"
        
        # If already in correct format or unrecognized format, return as is
        return allele
