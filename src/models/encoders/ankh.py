"""
Ankh HLA Encoder
----------------
Implementation of HLA encoder using Ankh protein language models.

Ankh is a newer protein language model designed specifically for protein analysis
(not adapted from NLP models). It offers efficient inference with competitive
performance, making it ideal for production deployments.
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
    from transformers import AutoTokenizer, AutoModel, logging as hf_logging
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
    # Suppress excessive warnings from transformers
    hf_logging.set_verbosity_error()
except ImportError:
    logger.warning(
        "Transformers library or huggingface_hub not installed; "
        "Ankh encoding not available. Try 'pip install transformers huggingface_hub'."
    )
    TRANSFORMERS_AVAILABLE = False

# Optional native Ankh package (pre-built loaders)
try:
    import ankh as ankh_models
    ANKH_PACKAGE_AVAILABLE = True
except ImportError:
    ankh_models = None
    ANKH_PACKAGE_AVAILABLE = False


class AnkhEncoder(HLAEncoder):
    """Ankh-based encoder for HLA alleles.
    
    Ankh is a protein language model specifically designed for protein analysis,
    offering efficient inference with competitive results. Unlike ProtBERT and ESM
    which were adapted from NLP models, Ankh was built from the ground up for proteins.
    
    Two model variants are available:
    - Ankh Base: ~50M parameters, very fast inference, good for production
    - Ankh Large: ~650M parameters, comparable to ESM-2 650M, better accuracy
    
    Attributes:
        model_name: Hugging Face model identifier
        device: Device to run model on ('cpu' or 'cuda')
        pooling_strategy: How to pool token embeddings ('mean' or 'cls')
        model: Ankh model instance
        tokenizer: Ankh tokenizer instance
        
    Example:
        >>> # Use base model for fast inference
        >>> encoder = AnkhEncoder("data/sequences.pkl", model_variant="base")
        >>> embedding = encoder.get_embedding("A*01:01")
        >>> embedding.shape
        (768,)
        >>> 
        >>> # Use large model for better accuracy
        >>> encoder = AnkhEncoder("data/sequences.pkl", model_variant="large")
        >>> embedding = encoder.get_embedding("A*01:01")
        >>> embedding.shape
        (1536,)
    """
    
    # Model variant configurations
    MODEL_VARIANTS = {
        "base": {
            "model_name": "ElnaggarLab/ankh-base",
            "embedding_dim": 768,
            "params": "50M",
            "description": "Fast inference, good accuracy",
            "package_loaders": ["load_ankh_base", "load_base_model"],
        },
        "large": {
            "model_name": "ElnaggarLab/ankh-large",
            "embedding_dim": 1536,
            "params": "650M",
            "description": "Better accuracy, comparable to ESM-2 650M",
            "package_loaders": ["load_ankh_large", "load_large_model"],
        },
    }
    
    def __init__(
        self,
        sequence_file: Union[str, Path],
        cache_dir: Union[str, Path] = "./data/embeddings/ankh",
        model_variant: str = "base",
        backend: str = "auto",
        model_name: Optional[str] = None,
        locus: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        verify_ssl: bool = True,
        hf_token: Optional[str] = None,
    ):
        """Initialize Ankh encoder.
        
        Args:
            sequence_file: Path to pickle file with HLA sequences
            cache_dir: Directory to cache embeddings (defaults to ./data/embeddings/ankh)
            model_variant: Model variant to use ('base' or 'large')
                - 'base': 50M params, 768-dim embeddings, fast
                - 'large': 650M params, 1536-dim embeddings, accurate
            backend: Backend to load models ('auto', 'huggingface', 'ankh')
                - 'auto': Try Hugging Face first, fall back to `ankh` package if available
                - 'huggingface': Force use of transformers/HF Hub
                - 'ankh': Force use of the pip `ankh` package loaders
            model_name: Optional custom Hugging Face model name.
                If provided, overrides model_variant.
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1')
            device: Device to run model on ('cpu', 'cuda', or None for auto-detection)
            pooling_strategy: How to pool token embeddings
                - 'mean': Average all token embeddings (recommended)
                - 'cls': Use [CLS] token embedding
            verify_ssl: Whether to verify SSL certificates when downloading models
            hf_token: Optional Hugging Face Hub token for authenticated downloads
            
        Raises:
            ImportError: If transformers library not installed
            ValueError: If model_variant is invalid
            RuntimeError: If model cannot be loaded
            
        Example:
            >>> # Fast inference with base model
            >>> encoder = AnkhEncoder("data/sequences.pkl", model_variant="base")
            >>> 
            >>> # Better accuracy with large model
            >>> encoder = AnkhEncoder(
            ...     "data/sequences.pkl",
            ...     model_variant="large",
            ...     device="cuda"
            ... )
            >>> 
            >>> # Custom model
            >>> encoder = AnkhEncoder(
            ...     "data/sequences.pkl",
            ...     model_name="ElnaggarLab/ankh-base"
            ... )
        """
        backend = (backend or "auto").lower()
        if backend not in {"auto", "huggingface", "ankh"}:
            raise ValueError("backend must be 'auto', 'huggingface', or 'ankh'")

        if backend == "huggingface" and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not installed; cannot load Ankh via Hugging Face. "
                "Install transformers or pick backend='ankh'."
            )
        if backend == "ankh" and not ANKH_PACKAGE_AVAILABLE:
            raise ImportError(
                "The 'ankh' package is not installed. Run 'pip install ankh' or use backend='huggingface'."
            )
        if backend == "auto" and not (TRANSFORMERS_AVAILABLE or ANKH_PACKAGE_AVAILABLE):
            raise ImportError(
                "Neither transformers nor the 'ankh' package are installed; install one of them to use AnkhEncoder."
            )

        # Validate model variant
        if model_name is None:
            if model_variant not in self.MODEL_VARIANTS:
                raise ValueError(
                    f"model_variant must be one of {list(self.MODEL_VARIANTS.keys())}, "
                    f"got '{model_variant}'"
                )
            model_name = self.MODEL_VARIANTS[model_variant]["model_name"]
            logger.info(
                f"Using Ankh {model_variant}: {self.MODEL_VARIANTS[model_variant]['description']}"
            )
        
        # Ensure cache dir is encoder-specific and variant-specific
        cache_dir = Path(cache_dir)
        if cache_dir.name != 'ankh':
            cache_dir = cache_dir / 'ankh'
        # Add variant subdirectory for better organization
        if model_name and "ankh" in model_name.lower():
            variant_suffix = "base" if "base" in model_name.lower() else "large"
            cache_dir = cache_dir / variant_suffix

        # Initialize base class
        super().__init__(sequence_file, cache_dir, locus, verify_ssl=verify_ssl)

        # Store configuration
        self.backend = backend
        self.active_backend: Optional[str] = None
        self._tokenizer_style = "string"
        self.model_name = model_name
        self.model_variant = model_variant
        self.pooling_strategy = pooling_strategy
        self.hf_token = hf_token

        # Validate pooling strategy
        if self.pooling_strategy not in ['mean', 'cls']:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'cls', got '{self.pooling_strategy}'"
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
        """Load Ankh model using the requested backend."""
        attempt_order = self._determine_backend_order()
        logger.info(
            "Initializing Ankh encoder (requested backend=%s, attempts=%s)",
            self.backend,
            attempt_order,
        )

        last_error: Optional[Exception] = None
        for backend in attempt_order:
            try:
                if backend == "huggingface":
                    self._load_model_huggingface()
                elif backend == "ankh":
                    self._load_model_with_native_package()
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unsupported backend '{backend}'")
                self.active_backend = backend
                logger.info(
                    "Ankh model loaded via %s backend on %s (variant=%s)",
                    backend,
                    self.device,
                    self.model_variant,
                )
                return
            except Exception as err:
                last_error = err
                logger.warning(
                    "Failed to load Ankh model via %s backend: %s", backend, err
                )
                if self.backend != "auto":
                    break

        backend_list = ", ".join(attempt_order)
        raise RuntimeError(
            f"Failed to load Ankh model using backends [{backend_list}]: {last_error}"
        )

    def _determine_backend_order(self) -> List[str]:
        if self.backend == "auto":
            order: List[str] = []
            if TRANSFORMERS_AVAILABLE:
                order.append("huggingface")
            if ANKH_PACKAGE_AVAILABLE:
                order.append("ankh")
            return order
        return [self.backend]

    def _load_model_huggingface(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available; install transformers to use the Hugging Face backend."
            )

        logger.info(
            "Loading Ankh model/tokenizer from Hugging Face: %s", self.model_name
        )

        # Attempt Hugging Face Hub login if token provided
        if self.hf_token:
            logger.info("Attempting Hugging Face Hub login with provided token...")
            try:
                login(token=self.hf_token)
                logger.info("Hugging Face Hub login successful.")
            except Exception as login_err:
                logger.warning(
                    "Hugging Face Hub login failed: %s. Proceeding without authentication.",
                    login_err,
                )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        self._tokenizer_style = "string"

    def _load_model_with_native_package(self) -> None:
        if not ANKH_PACKAGE_AVAILABLE:
            raise ImportError(
                "The optional 'ankh' package is not installed. Install it via 'pip install ankh'."
            )

        variant_config = self.MODEL_VARIANTS.get(self.model_variant)
        if not variant_config:
            raise ValueError(
                f"Variant '{self.model_variant}' is not supported by the native Ankh loaders."
            )

        loader_names = variant_config.get("package_loaders") or []
        loader = None
        for name in loader_names:
            if hasattr(ankh_models, name):
                loader = getattr(ankh_models, name)
                logger.info(
                    "Loading %s using 'ankh' package loader '%s'",
                    self.model_variant,
                    name,
                )
                break

        if loader is None:
            raise ValueError(
                f"Native Ankh loader not available for variant '{self.model_variant}' (checked {loader_names})."
            )

        model, tokenizer = loader()

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self._tokenizer_style = "characters"

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence using Ankh.
        
        Ankh tokenization is similar to BERT but designed for proteins.
        The outputs are pooled according to the pooling strategy.
        
        Args:
            sequence: Protein sequence string (amino acid sequence)
            
        Returns:
            Embedding vector as numpy array
            - Shape (768,) for Ankh Base
            - Shape (1536,) for Ankh Large
            
        Raises:
            ValueError: If sequence is empty or contains invalid characters
            RuntimeError: If encoding fails due to model errors
            
        Note:
            Invalid amino acids are replaced with 'X' (unknown) by the tokenizer.
        """
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"Sequence must be a non-empty string, got {type(sequence)}")
        
        # Clean sequence
        sequence_clean = sequence.replace(" ", "")
        
        if not sequence_clean:
            raise ValueError("Sequence is empty after cleaning")
        
        try:
            # Tokenize
            tokenizer_input = sequence_clean
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": False,
                "truncation": True,
                "max_length": 512,
            }
            if self._tokenizer_style == "characters":
                tokenizer_input = list(sequence_clean)
                tokenizer_kwargs["is_split_into_words"] = True
                tokenizer_kwargs.setdefault("add_special_tokens", True)

            inputs = self.tokenizer(
                tokenizer_input,
                **tokenizer_kwargs,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract hidden states (last layer)
            # Shape: (batch_size=1, seq_len, hidden_size)
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
            elif self.pooling_strategy == "cls":
                # Use [CLS] token embedding (first token)
                embedding = hidden_states[:, 0, :].squeeze(0)
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
        batch_size: int = 16,  # Larger default for smaller base model
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """Encode multiple HLA alleles efficiently in batches.
        
        Processes alleles in batches to improve GPU utilization and speed.
        Automatically handles caching and skips already-encoded alleles
        unless force=True.
        
        Args:
            alleles: List of HLA allele identifiers to encode
            batch_size: Number of sequences to process simultaneously
                Default is 16 for Ankh Base (larger than ProtBERT due to smaller model).
                For Ankh Large, consider reducing to 8 if encountering memory issues.
                Larger values increase speed but require more GPU memory.
            force: If True, regenerate embeddings even if cached
            
        Returns:
            Dictionary mapping allele identifiers to embedding vectors
            
        Raises:
            ValueError: If any allele cannot be resolved to a sequence
            RuntimeError: If batch encoding fails
            
        Example:
            >>> encoder = AnkhEncoder("data/sequences.pkl", model_variant="base")
            >>> alleles = ["A*01:01", "A*02:01", "B*07:02"]
            >>> embeddings = encoder.batch_encode_alleles(alleles, batch_size=32)
            >>> embeddings["A*01:01"].shape
            (768,)
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
        
        # Clean sequences
        sequences_clean = [seq.replace(" ", "") for seq in sequences]
        
        try:
            # Tokenize batch with padding
            tokenizer_input = sequences_clean
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": 512,
            }
            if self._tokenizer_style == "characters":
                tokenizer_input = [list(seq) for seq in sequences_clean]
                tokenizer_kwargs["is_split_into_words"] = True
                tokenizer_kwargs.setdefault("add_special_tokens", True)

            inputs = self.tokenizer(
                tokenizer_input,
                **tokenizer_kwargs,
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
            elif self.pooling_strategy == "cls":
                # Use [CLS] token embedding for each sequence
                embeddings = hidden_states[:, 0, :]
            
            # Convert to list of numpy arrays
            embeddings_np = [emb.cpu().numpy() for emb in embeddings]
            
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to batch encode {len(sequences)} sequences: {e}")
            raise RuntimeError(f"Failed to batch encode sequences: {e}") from e
