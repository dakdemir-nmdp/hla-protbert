"""
HLA Encoder Base Class
---------------------
Base class for HLA sequence encoders with common functionality.
"""
import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# Try to import pyard for HLA nomenclature
try:
    import pyard
    PYARD_AVAILABLE = True
except ImportError:
    logger.warning("pyard not installed; allele resolution mapping will be limited")
    PYARD_AVAILABLE = False

class HLAEncoder:
    """Base class for HLA encoders
    
    Provides common functionality for different encoder implementations:
    - Cache management for embeddings
    - Fallback mechanisms for allele resolution
    - Sequence retrieval and standardization
    
    Subclasses should implement _encode_sequence method.
    """
    
    def __init__(
        self, 
        sequence_file: Union[str, Path],
        cache_dir: Union[str, Path] = "./data/embeddings",
        locus: Optional[str] = None
    ):
        """Initialize encoder
        
        Args:
            sequence_file: Path to pickle file with HLA sequences
            cache_dir: Directory to cache embeddings
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1')
                   If provided, only alleles of this locus will be encoded
        """
        self.sequence_file = Path(sequence_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.locus = locus
        
        # Set cache file based on locus
        if locus:
            self.embedding_cache_file = self.cache_dir / f"hla_{locus}_embeddings.pkl"
        else:
            self.embedding_cache_file = self.cache_dir / "hla_embeddings.pkl"
            
        # Load sequences
        self._load_sequences()
        
        # Initialize ARD (if available)
        self._initialize_ard()
        
        # Load cached embeddings
        self.embeddings = self._load_embedding_cache()
    
    def _load_sequences(self) -> None:
        """Load HLA sequences from file"""
        if not self.sequence_file.exists():
            logger.error(f"Sequence file not found: {self.sequence_file}")
            raise FileNotFoundError(f"Sequence file not found: {self.sequence_file}")
            
        try:
            with open(self.sequence_file, 'rb') as f:
                self.sequences = pickle.load(f)
                
            logger.info(f"Loaded {len(self.sequences)} sequences from {self.sequence_file}")
            
            # Filter by locus if specified
            if self.locus:
                self.sequences = {
                    allele: seq for allele, seq in self.sequences.items() 
                    if allele.startswith(f"{self.locus}*")
                }
                logger.info(f"Filtered to {len(self.sequences)} {self.locus} sequences")
                
        except Exception as e:
            logger.error(f"Error loading sequences: {e}")
            raise
    
    def _initialize_ard(self) -> None:
        """Initialize Antigen Recognition Domain (ARD) mapper if available"""
        if PYARD_AVAILABLE:
            try:
                self.ard = pyard.ARD()
                logger.info("Initialized ARD mapper for allele resolution")
            except Exception as e:
                logger.warning(f"Failed to initialize ARD: {e}")
                self.ard = None
        else:
            self.ard = None
    
    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings from file
        
        Returns:
            Dict mapping allele names to embeddings
        """
        if self.embedding_cache_file.exists():
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} cached embeddings from {self.embedding_cache_file}")
                return embeddings
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
                return {}
        else:
            logger.info(f"No embedding cache found at {self.embedding_cache_file}")
            return {}
    
    def _save_embedding_cache(self) -> None:
        """Save embeddings to cache file"""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Saved {len(self.embeddings)} embeddings to {self.embedding_cache_file}")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    def get_sequence(self, allele: str) -> Optional[str]:
        """Get sequence for an allele with fallbacks
        
        Will try multiple resolution methods if the allele is not directly found:
        1. Direct lookup
        2. Resolution to 2-field (if more fields provided)
        3. ARD mapping (if pyard available)
        4. Resolution to 1-field
        
        Args:
            allele: HLA allele name
            
        Returns:
            Protein sequence or None if not found
        """
        # Standardize allele format
        allele = self._standardize_allele(allele)
        
        # Try direct lookup
        if allele in self.sequences:
            return self.sequences[allele]
        
        # Try to resolve to 2-field if more fields provided
        if ':' in allele and allele.count(':') > 1:
            two_field = ':'.join(allele.split(':')[:2])
            if two_field in self.sequences:
                logger.info(f"Resolved {allele} to {two_field}")
                return self.sequences[two_field]
        
        # Try ARD mapping if available
        if self.ard is not None:
            try:
                mapped = self.ard.redux_gl(allele, 'lgx')
                if mapped in self.sequences:
                    logger.info(f"ARD mapped {allele} to {mapped}")
                    return self.sequences[mapped]
            except Exception as e:
                logger.debug(f"ARD mapping failed for {allele}: {e}")
        
        # Try resolving to first field
        if ':' in allele:
            one_field = allele.split(':')[0]
            # Look for any allele with this first field
            for seq_allele in self.sequences:
                if seq_allele.startswith(f"{one_field}:"):
                    logger.info(f"Fell back from {allele} to {seq_allele}")
                    return self.sequences[seq_allele]
        
        logger.warning(f"No sequence found for allele {allele}")
        return None
    
    def _standardize_allele(self, allele: str) -> str:
        """Standardize allele name format
        
        Args:
            allele: HLA allele name
            
        Returns:
            Standardized allele name
        """
        # Remove HLA- prefix if present
        if allele.startswith('HLA-'):
            allele = allele[4:]
        
        # Handle format without '*'
        if '*' not in allele and self.locus:
            if allele.startswith(self.locus):
                # Format like A0101
                if len(allele) > len(self.locus) and allele[len(self.locus):].isdigit():
                    digits = allele[len(self.locus):]
                    if len(digits) == 4:  # 4-digit format like A0101
                        return f"{self.locus}*{digits[:2]}:{digits[2:]}"
            else:
                # Just digits, assume current locus
                if allele.isdigit() and len(allele) == 4:
                    return f"{self.locus}*{allele[:2]}:{allele[2:]}"
        
        return allele
    
    def get_embedding(self, allele: str) -> np.ndarray:
        """Get embedding for an allele
        
        If the embedding is cached, returns it directly.
        Otherwise, gets the sequence and encodes it.
        
        Args:
            allele: HLA allele name
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If no sequence found for allele
        """
        # Standardize allele name
        allele = self._standardize_allele(allele)
        
        # Check if embedding is cached
        if allele in self.embeddings:
            return self.embeddings[allele]
        
        # Get sequence
        sequence = self.get_sequence(allele)
        if sequence is None:
            raise ValueError(f"No sequence found for allele {allele}")
        
        # Encode sequence
        embedding = self._encode_sequence(sequence)
        
        # Cache embedding
        self.embeddings[allele] = embedding
        self._save_embedding_cache()
        
        return embedding
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence to a vector
        
        This method should be implemented by subclasses.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("Subclasses must implement _encode_sequence")
    
    def batch_encode_alleles(self, alleles: List[str]) -> Dict[str, np.ndarray]:
        """Encode multiple alleles in batch
        
        Args:
            alleles: List of HLA allele names
            
        Returns:
            Dict mapping allele names to embeddings
        """
        results = {}
        missing = []
        
        for allele in alleles:
            try:
                results[allele] = self.get_embedding(allele)
            except Exception as e:
                logger.warning(f"Error encoding {allele}: {e}")
                missing.append(allele)
        
        if missing:
            logger.warning(f"Failed to encode {len(missing)} alleles: {missing}")
            
        return results
    
    def find_similar_alleles(
        self, 
        allele: str, 
        top_k: int = 5, 
        metric: str = 'cosine'
    ) -> List[Tuple[str, float]]:
        """Find most similar alleles to the given allele
        
        Args:
            allele: Query HLA allele
            top_k: Number of similar alleles to return
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            List of (allele_name, similarity_score) tuples
        """
        # Get query embedding
        try:
            query_embedding = self.get_embedding(allele)
        except Exception as e:
            logger.error(f"Error getting embedding for {allele}: {e}")
            return []
        
        # Compute similarities
        similarities = []
        for other_allele, other_embedding in self.embeddings.items():
            if other_allele == allele:
                continue
                
            if metric == 'cosine':
                similarity = self._cosine_similarity(query_embedding, other_embedding)
            elif metric == 'euclidean':
                similarity = -self._euclidean_distance(query_embedding, other_embedding)
            elif metric == 'manhattan':
                similarity = -self._manhattan_distance(query_embedding, other_embedding)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            similarities.append((other_allele, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors
        
        Args:
            a, b: Input vectors
            
        Returns:
            Cosine similarity (1 = identical, 0 = orthogonal)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors
        
        Args:
            a, b: Input vectors
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(a - b)
    
    @staticmethod
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Manhattan distance between two vectors
        
        Args:
            a, b: Input vectors
            
        Returns:
            Manhattan distance
        """
        return np.sum(np.abs(a - b))
