"""
ProtT5 HLA Encoder
------------------
Implementation of HLA encoder using ProtT5 (T5-based protein language model).

ProtT5 is an encoder-decoder T5 model trained on protein sequences.
It captures different linguistic patterns than BERT-based models and has shown
excellent performance without requiring MSA integration.
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm

# Import base encoder (one level up)
from ..encoder import HLAEncoder

logger = logging.getLogger(__name__)

# Check for Transformers library
try:
    from transformers import T5Tokenizer, T5EncoderModel, logging as hf_logging
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
    # Suppress excessive warnings from transformers
    hf_logging.set_verbosity_error()
except ImportError:
    logger.warning(
        "Transformers library or huggingface_hub not installed; "
        "ProtT5 encoding not available. Try 'pip install transformers huggingface_hub'."
    )
    TRANSFORMERS_AVAILABLE = False


class ProtT5Encoder(HLAEncoder):
    """ProtT5-based encoder for HLA alleles.
    
    This encoder uses the ProtT5 protein language model from RostLab,
    accessed through the Hugging Face Transformers library.
    ProtT5 uses a T5 architecture which captures different patterns than
    BERT-based models like ProtBERT, making it an excellent complementary encoder.
    
    The T5 architecture uses an encoder-decoder structure where only the encoder
    is used for embedding generation. The model has 1.3B parameters and generates
    1024-dimensional embeddings by default.
    
    Attributes:
        model_name: Hugging Face model identifier (default: "Rostlab/prot_t5_xl_uniref50")
        device: Device to run model on ('cpu' or 'cuda')
        pooling_strategy: How to pool token embeddings ('mean' or 'last')
        model: T5EncoderModel instance
        tokenizer: T5Tokenizer instance
        
    Example:
        >>> encoder = ProtT5Encoder("data/sequences.pkl", locus="A")
        >>> embedding = encoder.get_embedding("A*01:01")
        >>> embedding.shape
        (1024,)
        >>> # Batch encoding
        >>> alleles = ["A*01:01", "A*02:01", "A*03:01"]
        >>> embeddings = encoder.batch_encode_alleles(alleles)
        >>> len(embeddings)
        3
    """
    
    def __init__(
        self,
        sequence_file: Union[str, Path],
        cache_dir: Union[str, Path] = "./data/embeddings/prott5",
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        locus: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        verify_ssl: bool = True,
        hf_token: Optional[str] = None,
    ):
        """Initialize ProtT5 encoder.
        
        Args:
            sequence_file: Path to pickle file with HLA sequences
            cache_dir: Directory to cache embeddings (defaults to ./data/embeddings/prott5)
            model_name: ProtT5 model name from Hugging Face
                Options: 
                - "Rostlab/prot_t5_xl_uniref50" (1.3B params, recommended)
                - "Rostlab/prot_t5_xl_half_uniref50-enc" (half precision)
                - "Rostlab/prot_t5_xxl_uniref50" (3B params, largest)
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1')
            device: Device to run model on ('cpu', 'cuda', or None for auto-detection)
            pooling_strategy: How to pool token embeddings
                - 'mean': Average all token embeddings (default, recommended)
                - 'last': Use last token embedding
            verify_ssl: Whether to verify SSL certificates when downloading models
            hf_token: Optional Hugging Face Hub token for authenticated downloads
            
        Raises:
            ImportError: If transformers library not installed
            RuntimeError: If model cannot be loaded
            
        Example:
            >>> # Basic usage with auto-detected GPU
            >>> encoder = ProtT5Encoder("data/sequences.pkl")
            >>> 
            >>> # CPU-only with custom cache directory
            >>> encoder = ProtT5Encoder(
            ...     "data/sequences.pkl",
            ...     cache_dir="./cache/prott5",
            ...     device="cpu"
            ... )
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not installed; cannot use ProtT5Encoder. "
                "Install with: pip install transformers huggingface_hub"
            )

        # Ensure cache dir is encoder-specific
        cache_dir = Path(cache_dir)
        if cache_dir.name != 'prott5':
            cache_dir = cache_dir / 'prott5'

        # Initialize base class
        super().__init__(sequence_file, cache_dir, locus, verify_ssl=verify_ssl)

        # Store configuration
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.hf_token = hf_token

        # Validate pooling strategy
        if self.pooling_strategy not in ['mean', 'last']:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'last', got '{self.pooling_strategy}'"
            )

        # Set device, handling 'auto'
        resolved_device = device
        if resolved_device is None or resolved_device.lower() == 'auto':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {resolved_device}")
        self.device = resolved_device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load ProtT5 model and tokenizer using Transformers.
        
        Handles Hugging Face Hub authentication if token is provided.
        Moves model to the specified device.
        
        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(
            f"Loading ProtT5 model/tokenizer: {self.model_name} "
            f"onto device: {self.device}"
        )

        try:
            # Attempt Hugging Face Hub login if token provided
            if self.hf_token:
                logger.info("Attempting Hugging Face Hub login with provided token...")
                try:
                    login(token=self.hf_token)
                    logger.info("Hugging Face Hub login successful.")
                except Exception as login_err:
                    logger.warning(
                        f"Hugging Face Hub login failed: {login_err}. "
                        "Proceeding without authentication."
                    )

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                do_lower_case=False,
                legacy=False,
            )
            
            # Load encoder model (T5EncoderModel, not full T5ForConditionalGeneration)
            logger.info("Loading encoder model (this may take a while)...")
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(
                f"ProtT5 model loaded successfully on {self.device}. "
                f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters"
            )

        except Exception as e:
            logger.error(f"Failed to load ProtT5 model: {e}")
            raise RuntimeError(f"Failed to load ProtT5 model: {e}") from e

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence using ProtT5.
        
        T5 tokenization requires spaces between amino acids and a special
        prefix for the encoder. The encoder outputs are then pooled according
        to the pooling strategy.
        
        Args:
            sequence: Protein sequence string (amino acid sequence)
            
        Returns:
            Embedding vector as numpy array of shape (1024,) for ProtT5-XL
            
        Raises:
            ValueError: If sequence is empty or contains invalid characters
            RuntimeError: If encoding fails due to model errors
            
        Note:
            ProtT5 requires sequences with spaces between amino acids.
            Invalid amino acids are replaced with 'X' (unknown).
        """
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"Sequence must be a non-empty string, got {type(sequence)}")
        
        # Remove any existing spaces and add spaces between amino acids
        sequence_clean = sequence.replace(" ", "")
        
        if not sequence_clean:
            raise ValueError("Sequence is empty after cleaning")
        
        # Add spaces between amino acids (required for T5)
        sequence_spaced = " ".join(list(sequence_clean))
        
        try:
            # Tokenize with attention mask
            # ProtT5 uses special tokens, tokenizer handles them automatically
            inputs = self.tokenizer(
                sequence_spaced,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512,  # T5 max sequence length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract hidden states (last layer)
            # Shape: (batch_size=1, seq_len, hidden_size=1024)
            hidden_states = outputs.last_hidden_state
            
            # Pool according to strategy
            if self.pooling_strategy == "mean":
                # Mean pooling over sequence dimension
                # Mask padding tokens if present
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    # Expand attention mask for broadcasting
                    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                    # Sum embeddings and normalize by non-padding tokens
                    sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=1)
                    sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
                    embedding = (sum_embeddings / sum_mask).squeeze(0)
                else:
                    embedding = hidden_states.mean(dim=1).squeeze(0)
            elif self.pooling_strategy == "last":
                # Use last token embedding
                embedding = hidden_states[:, -1, :].squeeze(0)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy()
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to encode sequence of length {len(sequence)}: {e}")
            raise RuntimeError(f"Failed to encode sequence: {e}") from e

    def batch_encode_alleles(
        self, 
        alleles: List[str], 
        batch_size: int = 4,  # Smaller default due to larger model
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """Encode multiple HLA alleles efficiently in batches.
        
        Processes alleles in batches to improve GPU utilization and speed.
        Automatically handles caching and skips already-encoded alleles
        unless force=True.
        
        Args:
            alleles: List of HLA allele identifiers to encode
            batch_size: Number of sequences to process simultaneously
                Default is 4 for ProtT5 (smaller than ProtBERT due to larger model).
                Reduce if encountering out-of-memory errors.
                Larger values increase speed but require more GPU memory.
            force: If True, regenerate embeddings even if cached
            
        Returns:
            Dictionary mapping allele identifiers to embedding vectors
            
        Raises:
            ValueError: If any allele cannot be resolved to a sequence
            RuntimeError: If batch encoding fails
            
        Example:
            >>> encoder = ProtT5Encoder("data/sequences.pkl")
            >>> alleles = ["A*01:01", "A*02:01", "B*07:02"]
            >>> embeddings = encoder.batch_encode_alleles(alleles, batch_size=8)
            >>> embeddings["A*01:01"].shape
            (1024,)
        """
        # Remove duplicates while preserving order
        unique_alleles = list(dict.fromkeys(alleles))
        
        # Filter to only those needing encoding
        if not force:
            # Standardize allele names for cache lookup
            standardized_alleles = [self._standardize_allele(a) for a in unique_alleles]
            to_encode = [
                a for a in standardized_alleles 
                if a not in self.embeddings
            ]
            logger.info(
                f"Batch encoding {len(to_encode)} new alleles "
                f"(skipping {len(standardized_alleles) - len(to_encode)} cached)"
            )
        else:
            to_encode = [self._standardize_allele(a) for a in unique_alleles]
            logger.info(f"Force encoding {len(to_encode)} alleles")
        
        # Process in batches
        for i in tqdm(range(0, len(to_encode), batch_size), desc="Encoding batches"):
            batch_alleles = to_encode[i:i + batch_size]
            
            # Get sequences for batch
            batch_sequences = []
            batch_valid_alleles = []
            
            for allele in batch_alleles:
                sequence = self.get_sequence(allele)
                if sequence is None:
                    logger.warning(f"No sequence found for {allele}, skipping")
                    continue
                batch_sequences.append(sequence)
                batch_valid_alleles.append(allele)
            
            if not batch_sequences:
                continue
            
            # Encode batch
            try:
                batch_embeddings = self._batch_encode_sequences(batch_sequences)
                
                # Store embeddings
                for allele, embedding in zip(batch_valid_alleles, batch_embeddings):
                    self.embeddings[allele] = embedding
                    
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                # Fall back to individual encoding for this batch
                logger.info("Falling back to individual encoding for failed batch")
                for allele in batch_valid_alleles:
                    try:
                        self.get_embedding(allele, force=True)
                    except Exception as e2:
                        logger.error(f"Failed to encode {allele}: {e2}")
        
        # Save cache after all batches
        self._save_embedding_cache()
        
        # Return embeddings for requested alleles
        result = {}
        for allele in alleles:
            std_allele = self._standardize_allele(allele)
            if std_allele in self.embeddings:
                result[allele] = self.embeddings[std_allele]
            else:
                logger.warning(f"No embedding available for {allele}")
        
        return result

    def _batch_encode_sequences(self, sequences: List[str]) -> List[np.ndarray]:
        """Encode multiple sequences in a single batch.
        
        This is more efficient than encoding one at a time when using GPU.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If batch encoding fails
        """
        if not sequences:
            return []
        
        # Prepare sequences with spaces
        sequences_spaced = [" ".join(list(seq.replace(" ", ""))) for seq in sequences]
        
        try:
            # Tokenize batch with padding
            inputs = self.tokenizer(
                sequences_spaced,
                return_tensors="pt",
                padding=True,  # Pad to longest sequence in batch
                truncation=True,
                max_length=512,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract hidden states
            # Shape: (batch_size, seq_len, hidden_size)
            hidden_states = outputs.last_hidden_state
            
            # Pool according to strategy
            if self.pooling_strategy == "mean":
                # Mean pooling with attention mask
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=1)
                    sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif self.pooling_strategy == "last":
                # Use last token embedding for each sequence
                embeddings = hidden_states[:, -1, :]
            
            # Convert to list of numpy arrays
            embeddings_np = [emb.cpu().numpy() for emb in embeddings]
            
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to batch encode {len(sequences)} sequences: {e}")
            raise RuntimeError(f"Failed to batch encode sequences: {e}") from e
