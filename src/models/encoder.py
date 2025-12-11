"""
HLA Encoder Base Class
---------------------
Base class for HLA sequence encoders with common functionality.

This module provides the abstract base class for all HLA encoders in the system.
All encoder implementations (ProtBERT, ESM, etc.) must inherit from this class
and implement the required abstract methods.
"""
import os
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    from huggingface_hub import configure_http_backend
    HF_HUB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    configure_http_backend = None
    HF_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import pyard for HLA nomenclature
try:
    from py_ard import ARD
    PYARD_AVAILABLE = True
except ImportError:
    logger.warning("py-ard not installed; allele resolution mapping will be limited")
    PYARD_AVAILABLE = False


class HLAEncoder(ABC):
    """Abstract base class for HLA allele encoders.
    
    Provides common functionality for different encoder implementations:
    - Cache management for embeddings
    - Fallback mechanisms for allele resolution
    - Sequence retrieval and standardization
    - Similarity search
    
    Subclasses must implement the abstract method:
        - _encode_sequence: Convert a protein sequence to an embedding vector
    
    Attributes:
        sequence_file: Path to pickle file containing HLA sequences
        cache_dir: Directory for caching embeddings
        locus: Optional HLA locus filter (e.g., 'A', 'B', 'DRB1')
        verify_ssl: Whether to verify SSL certificates
        sequences: Dictionary mapping allele names to protein sequences
        embeddings: Dictionary mapping allele names to embedding vectors
        ard: Optional ARD (Allele Resolution) mapper instance
        
    Example:
        >>> # Subclass implementation
        >>> class MyEncoder(HLAEncoder):
        ...     def _encode_sequence(self, sequence: str) -> np.ndarray:
        ...         # Custom encoding logic
        ...         return np.random.rand(768)
        >>> 
        >>> encoder = MyEncoder("data/sequences.pkl")
        >>> embedding = encoder.get_embedding("A*01:01")
    """
    
    _current_ssl_verification: Optional[bool] = None

    def __init__(
        self, 
        sequence_file: Union[str, Path],
        cache_dir: Union[str, Path] = "./data/embeddings",
        locus: Optional[str] = None,
        verify_ssl: bool = True
    ) -> None:
        """Initialize encoder with sequence data and configuration.
        
        Args:
            sequence_file: Path to pickle file with HLA sequences. Must exist and contain
                a dictionary mapping allele names to protein sequences.
            cache_dir: Directory to cache embeddings. Will be created if it doesn't exist.
                Defaults to "./data/embeddings".
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1'). If provided, only alleles
                of this locus will be encoded. None means all loci will be processed.
            verify_ssl: Whether to verify SSL certificates when downloading models.
                Set to False for systems with SSL issues.
                
        Raises:
            TypeError: If sequence_file or cache_dir are not string or Path objects
            FileNotFoundError: If sequence_file does not exist
            ValueError: If locus is provided but invalid format
        """
        # Validate input types
        if not isinstance(sequence_file, (str, Path)):
            raise TypeError(f"sequence_file must be str or Path, got {type(sequence_file)}")
        if not isinstance(cache_dir, (str, Path)):
            raise TypeError(f"cache_dir must be str or Path, got {type(cache_dir)}")
        if locus is not None and not isinstance(locus, str):
            raise TypeError(f"locus must be str or None, got {type(locus)}")
        self.sequence_file = Path(sequence_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.locus = locus
        self.verify_ssl = bool(verify_ssl)
        self._configure_ssl_verification()
        
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

    def _configure_ssl_verification(self) -> None:
        """Configure SSL verification for Hugging Face Hub downloads."""
        cls = self.__class__
        desired_state = bool(self.verify_ssl)

        # Default state is already verified; no need to reconfigure until toggled
        if desired_state and cls._current_ssl_verification is None:
            cls._current_ssl_verification = True
            return

        if cls._current_ssl_verification == desired_state:
            return

        if not HF_HUB_AVAILABLE:
            if not desired_state:
                logger.warning(
                    "huggingface_hub is not installed; cannot disable SSL verification. "
                    "Install huggingface_hub or keep verify_ssl=True."
                )
            return

        if desired_state:
            configure_http_backend()
            logger.info("Re-enabled SSL verification for Hugging Face Hub downloads")
        else:
            import requests
            try:
                import urllib3
            except ImportError:  # pragma: no cover - urllib3 is a requests dependency
                urllib3 = None

            def insecure_backend_factory():
                session = requests.Session()
                session.verify = False
                session.trust_env = True
                return session

            configure_http_backend(backend_factory=insecure_backend_factory)
            if urllib3 is not None:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning(
                "Disabled SSL verification for Hugging Face Hub downloads. "
                "Only use this option on trusted networks."
            )

        cls._current_ssl_verification = desired_state
    
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
        """Standardize allele name format with locus-specific inference
        
        Args:
            allele: HLA allele name
            
        Returns:
            Standardized allele name
            
        Note:
            Uses HLASequenceUtils.standardize_allele_name() for standard formats.
            When encoder is bound to a specific locus (self.locus is set), can infer
            locus from pure digit formats (e.g., '0101' -> 'A*01:01' if locus='A').
            
            WARNING: Only use digit-only formats in locus-specific contexts to avoid
            ambiguity (A0101 could be A*01:01 or A*10:10:1).
        """
        from src.data.sequence_utils import HLASequenceUtils
        
        # Handle pure digit formats ONLY when locus is explicitly known
        # This is safe because locus context eliminates ambiguity
        if self.locus and allele.isdigit() and len(allele) == 4:
            return f"{self.locus}*{allele[:2]}:{allele[2:]}"
        
        # Delegate to utility function for all standard IMGT/HLA formats
        return HLASequenceUtils.standardize_allele_name(allele)
    
    def get_embedding(self, allele: str, force: bool = False) -> np.ndarray:
        """Get embedding vector for a single HLA allele.
        
        Retrieves cached embedding if available and force=False. Otherwise,
        fetches the protein sequence and generates a new embedding using
        the encoder's model. Automatically caches new embeddings.
        
        Args:
            allele: HLA allele identifier (e.g., "A*01:01", "B*07:02").
                Accepts multiple formats and applies automatic standardization.
            force: If True, regenerate embedding even if already cached.
                Useful for testing or when encoder parameters have changed.
            
        Returns:
            Embedding vector as numpy array of shape (embedding_dim,).
            Typically 768 for ProtBERT or 1280 for ESM-2.
            
        Raises:
            TypeError: If allele is not a string
            ValueError: If no sequence found for the allele after trying all
                resolution fallbacks (2-field, ARD mapping, 1-field)
            RuntimeError: If encoding fails due to model errors
            
        Example:
            >>> encoder = ProtBERTEncoder("data/sequences.pkl")
            >>> embedding = encoder.get_embedding("A*01:01")
            >>> embedding.shape
            (768,)
            >>> # Force regeneration
            >>> embedding = encoder.get_embedding("A*01:01", force=True)
        """
        # Validate input type
        if not isinstance(allele, str):
            raise TypeError(f"allele must be string, got {type(allele).__name__}")
        # Standardize allele name
        allele = self._standardize_allele(allele)
        
        # Check if embedding is cached and not forcing regeneration
        if not force and allele in self.embeddings:
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
    
    @abstractmethod
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence to an embedding vector.
        
        This is the core encoding method that must be implemented by all subclasses.
        Each encoder (ProtBERT, ESM, etc.) implements its own encoding strategy.
        
        Args:
            sequence: Protein sequence string (amino acid sequence)
            
        Returns:
            Embedding vector as numpy array of shape (embedding_dim,)
            
        Raises:
            ValueError: If sequence is empty or contains invalid characters
            RuntimeError: If encoding fails due to model errors
            
        Note:
            Subclasses should handle model-specific preprocessing, tokenization,
            and pooling strategies within this method.
        """
        pass
    
    def batch_encode_alleles(
        self, 
        alleles: List[str], 
        batch_size: int = 8, 
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """Encode multiple HLA alleles efficiently with optional batching.
        
        Default implementation iterates over get_embedding(). Subclasses should
        override this for true batch processing when their models support it
        (e.g., ESM, ProtBERT with GPU).
        
        Args:
            alleles: List of HLA allele identifiers to encode.
                Can include duplicates (will be encoded once).
            batch_size: Number of sequences to process simultaneously.
                Only used by subclasses with true batch encoding.
                Larger values increase speed but require more memory.
            force: If True, regenerate embeddings even if cached.
                Applies to all alleles in the list.
            
        Returns:
            Dictionary mapping allele identifiers to embedding vectors.
            Failed alleles are omitted (not included in result).
            
        Raises:
            TypeError: If alleles is not a list or contains non-strings
            ValueError: If batch_size < 1
            
        Example:
            >>> encoder = ProtBERTEncoder("data/sequences.pkl")
            >>> alleles = ["A*01:01", "A*02:01", "B*07:02"]
            >>> embeddings = encoder.batch_encode_alleles(alleles, batch_size=16)
            >>> len(embeddings)
            3
            >>> embeddings["A*01:01"].shape
            (768,)
            
        Note:
            Progress bar displayed automatically for >100 alleles.
            Failed encodings logged as warnings but don't stop processing.
        """
        # Validate inputs
        if not isinstance(alleles, list):
            raise TypeError(f"alleles must be list, got {type(alleles).__name__}")
        if not all(isinstance(a, str) for a in alleles):
            raise TypeError("All alleles must be strings")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # Default implementation iterates over get_embedding.
        # Subclasses (like ESMEncoder) should override this for true batching.
        results = {}
        missing = []
        
        # Import tqdm here to avoid circular imports
        from tqdm import tqdm
        
        # Use tqdm for progress if many alleles
        allele_iterator = tqdm(alleles, desc="Encoding Alleles") if len(alleles) > 100 else alleles

        for allele in allele_iterator:
            try:
                # Pass force flag to get_embedding
                results[allele] = self.get_embedding(allele, force=force)
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
        """Find most similar alleles to a query allele based on embedding similarity.
        
        Computes similarity between query allele and all cached embeddings,
        returning the top-k most similar matches. Useful for identifying
        functionally similar alleles or potential cross-reactivity.
        
        Args:
            allele: Query HLA allele identifier (e.g., "A*01:01").
            top_k: Number of most similar alleles to return.
                Must be positive. If more than available alleles, returns all.
            metric: Distance/similarity metric to use:
                - 'cosine': Cosine similarity (1=identical, 0=orthogonal, -1=opposite)
                - 'euclidean': Negative Euclidean distance (higher=more similar)
                - 'manhattan': Negative Manhattan/L1 distance (higher=more similar)
            
        Returns:
            List of (allele_name, similarity_score) tuples, sorted by
            similarity descending. Query allele excluded from results.
            Empty list if query allele embedding cannot be generated.
            
        Raises:
            TypeError: If allele is not string or top_k not int
            ValueError: If metric not in supported metrics or top_k < 1
            
        Example:
            >>> encoder = ProtBERTEncoder("data/sequences.pkl")
            >>> # Find alleles similar to A*01:01
            >>> similar = encoder.find_similar_alleles("A*01:01", top_k=3)
            >>> for allele, score in similar:
            ...     print(f"{allele}: {score:.3f}")
            A*01:02: 0.995
            A*01:03: 0.987
            A*01:04: 0.981
            
        Note:
            Requires embeddings to already be cached. Run batch_encode_alleles
            first to populate cache if needed.
        """
        # Validate inputs
        if not isinstance(allele, str):
            raise TypeError(f"allele must be string, got {type(allele).__name__}")
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if metric not in ('cosine', 'euclidean', 'manhattan'):
            raise ValueError(f"metric must be 'cosine', 'euclidean', or 'manhattan', got '{metric}'")
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
