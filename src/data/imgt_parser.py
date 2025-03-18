"""
IMGT/HLA Database Parser
-----------------------
Tools to parse IMGT/HLA database files and extract protein sequences.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import BioPython's SeqIO for FASTA parsing
try:
    from Bio import SeqIO
except ImportError:
    logging.warning("BioPython not installed; some functionality may be limited")
    SeqIO = None

logger = logging.getLogger(__name__)

class IMGTParser:
    """Parser for IMGT/HLA database files"""
    
    def __init__(self, imgt_dir='./data/raw', output_dir='./data/processed'):
        """Initialize parser with input and output directories
        
        Args:
            imgt_dir: Directory containing raw IMGT/HLA data
            output_dir: Directory to store processed data
        """
        self.imgt_dir = Path(imgt_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if BioPython is available
        if SeqIO is None:
            logger.warning("BioPython not available; using fallback parser")
            self._parse_fasta = self._parse_fasta_fallback
        else:
            self._parse_fasta = self._parse_fasta_biopython
    
    def parse_protein_sequences(self) -> Dict[str, str]:
        """Parse protein sequences from IMGT/HLA FASTA files
        
        Returns:
            Dict mapping HLA allele names to their protein sequences
        """
        sequences = {}
        
        # Parse consolidated protein file if available
        hla_prot_file = self.imgt_dir / "hla_prot.fasta"
        if hla_prot_file.exists():
            logger.info(f"Parsing consolidated protein file: {hla_prot_file}")
            sequences.update(self._parse_fasta(hla_prot_file))
        else:
            # Parse individual locus files
            fasta_dir = self.imgt_dir / "fasta"
            if not fasta_dir.exists():
                logger.error(f"FASTA directory not found: {fasta_dir}")
                raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")
                
            logger.info(f"Parsing individual locus files from: {fasta_dir}")
            for fasta_file in fasta_dir.glob("*_prot.fasta"):
                logger.info(f"  Parsing: {fasta_file.name}")
                locus_sequences = self._parse_fasta(fasta_file)
                sequences.update(locus_sequences)
                logger.info(f"  Added {len(locus_sequences)} sequences from {fasta_file.name}")
        
        logger.info(f"Total sequences parsed: {len(sequences)}")
        
        # Save processed sequences
        output_file = self.output_dir / "hla_sequences.pkl"
        logger.info(f"Saving sequences to: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
            
        # Also save in plain text format
        txt_output = self.output_dir / "hla_sequences.txt"
        with open(txt_output, 'w') as f:
            for allele, seq in sorted(sequences.items()):
                f.write(f">{allele}\n{seq}\n")
        
        return sequences
    
    def _parse_fasta_biopython(self, fasta_file: Path) -> Dict[str, str]:
        """Parse FASTA file using BioPython
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Dict mapping HLA allele names to sequences
        """
        sequences = {}
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            allele_name = self._extract_allele_name(record.description)
            sequences[allele_name] = str(record.seq)
        return sequences
    
    def _parse_fasta_fallback(self, fasta_file: Path) -> Dict[str, str]:
        """Fallback FASTA parser for when BioPython is not available
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Dict mapping HLA allele names to sequences
        """
        sequences = {}
        current_allele = None
        current_sequence = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_allele and current_sequence:
                        sequences[current_allele] = ''.join(current_sequence)
                    
                    # Start new sequence
                    current_allele = self._extract_allele_name(line[1:])
                    current_sequence = []
                else:
                    # Continue building current sequence
                    current_sequence.append(line)
        
        # Save last sequence
        if current_allele and current_sequence:
            sequences[current_allele] = ''.join(current_sequence)
            
        return sequences
    
    def _extract_allele_name(self, description: str) -> str:
        """Extract standardized HLA allele name from FASTA description
        
        Args:
            description: Description line from FASTA file
            
        Returns:
            Standardized HLA allele name (e.g., 'A*01:01')
        """
        # Handle different formats in IMGT/HLA FASTA files
        # Format examples:
        # >HLA:HLA00001 A*01:01:01:01 1104 bp
        # >A*01:01:01:01
        parts = description.split()
        
        # Try to find the part with allele notation (containing *)
        for part in parts:
            if '*' in part:
                # Extract standardized form (e.g., first two fields)
                fields = part.split(':')
                if len(fields) >= 2:
                    return ':'.join([fields[0], fields[1]])
                return part
        
        # If no allele notation found, try other approaches
        if 'HLA' in description:
            # Try to extract from HLA:HLAxxxxx format
            for part in parts:
                if part.startswith('HLA:HLA'):
                    # The next part should be the allele
                    idx = parts.index(part)
                    if idx + 1 < len(parts) and '*' in parts[idx + 1]:
                        fields = parts[idx + 1].split(':')
                        if len(fields) >= 2:
                            return ':'.join([fields[0], fields[1]])
                        return parts[idx + 1]
        
        # Last resort: just use the whole description
        logger.warning(f"Could not extract allele name from: {description}")
        return description.split()[0]
