"""
ESM HLA Encoder
-------------------
Implementation of HLA encoder using ESM.
"""
import logging
import os # Import os module
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import logging # Keep logging import

# Import base encoder (one level up)
from ..encoder import HLAEncoder

logger = logging.getLogger(__name__)

# Check for Transformers library
try:
    from transformers import AutoTokenizer, AutoModel, logging as hf_logging
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
    # Suppress excessive warnings from transformers about loading weights
    hf_logging.set_verbosity_error()
except ImportError:
    logger.warning("Transformers library or huggingface_hub not installed; ESM encoding via Transformers not available. Try 'pip install transformers huggingface_hub'.")
    TRANSFORMERS_AVAILABLE = False


class ESMEncoder(HLAEncoder): # Renamed class
    """ESM-based encoder for HLA alleles.
    
    This encoder uses the ESM (Evolutionary Scale Modeling) protein language models
    from Meta AI, accessed through the Hugging Face Transformers library.
    It supports various ESM models including ESM-2 and others.
    """

    def __init__(
        self,
        sequence_file: Union[str, Path],
        cache_dir: Union[str, Path] = "./data/embeddings/esm", # Default to a specific subdir 'esm'
        model_name: str = "facebook/esm2_t33_650M_UR50D", # Changed default model
        locus: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        verify_ssl: bool = True,
        hf_token: Optional[str] = None, # Add parameter for HF token
        # verify_ssl is no longer needed as transformers handles it
    ):
        """Initialize ESM encoder using Transformers library

        Args:
            sequence_file: Path to pickle file with HLA sequences
            cache_dir: Directory to cache embeddings (defaults to ./data/embeddings/esm)
            model_name: ESM model name from Hugging Face (e.g., 'facebook/esm2_t33_650M_UR50D')
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1')
            device: Device to run model on ('cpu', 'cuda', or None for auto-detection)
            pooling_strategy: How to pool token embeddings ('mean', 'cls')
            hf_token: Optional Hugging Face Hub token for authenticated downloads
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not installed; cannot use ESMEncoder") # Updated class name in error

        # Modify cache dir to be encoder-specific
        cache_dir = Path(cache_dir)
        if not cache_dir.name == 'esm': # Ensure the final part is 'esm'
             cache_dir = cache_dir / 'esm'

        # Initialize base class - verify_ssl is removed
        super().__init__(sequence_file, cache_dir, locus)

        # Store model name (should be HF identifier now)
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.hf_token = hf_token # Store the token

        # Set device, handling 'auto'
        resolved_device = device
        if resolved_device is None or resolved_device.lower() == 'auto':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {resolved_device}")
        self.device = resolved_device # Store the resolved device ('cuda' or 'cpu')

        # Load model using Transformers
        self._load_model()

    def _load_model(self):
        """Load ESM model and tokenizer using Transformers""" # Updated docstring
        logger.info(f"Loading ESM model/tokenizer: {self.model_name} onto device: {self.device} using Transformers")

        try:
            # Attempt Hugging Face Hub login if library is available and token provided
            if self.hf_token:
                logger.info("Attempting Hugging Face Hub login with provided token...")
                try:
                    login(token=self.hf_token)
                    logger.info("Hugging Face Hub login successful or token already available.")
                except Exception as login_err:
                    # Log error but proceed, maybe model is public or cached
                    logger.warning(f"Hugging Face Hub login failed: {login_err}. Proceeding with model load attempt...")

            # Load tokenizer and model using AutoClasses
            # trust_remote_code=True might be needed for some ESM models if they have custom code
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, token=self.hf_token, trust_remote_code=True)

            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded ESM model '{self.model_name}' on {self.device} using Transformers") # Updated log message

        except Exception as e:
            logger.error(f"Error loading ESM model via Transformers: {e}") # Updated log message
            logger.error("Please ensure the model name is a valid Hugging Face identifier (e.g., 'facebook/esm2_t33_650M_UR50D')") # Updated example
            raise

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence using ESM via Transformers

        Args:
            sequence: Protein sequence

        Returns:
            Embedding vector
        """
        # Tokenize the sequence
        # Add spaces between residues for ESM tokenizers
        sequence_spaced = " ".join(list(sequence))
        inputs = self.tokenizer(sequence_spaced, return_tensors="pt", padding=False, truncation=False) # No padding/truncation for single sequence
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract last hidden state
            token_embeddings = outputs.last_hidden_state # Shape: (1, seq_len, hidden_dim)

        # Remove BOS/EOS tokens for pooling if they exist (ESM tokenizers usually add them)
        # Check tokenizer config or typical ESM behavior
        has_bos = self.tokenizer.bos_token_id is not None
        has_eos = self.tokenizer.eos_token_id is not None
        start_idx = 1 if has_bos else 0
        end_idx = -1 if has_eos else None
        token_embeddings_no_special = token_embeddings[:, start_idx:end_idx, :]

        # Pool token embeddings based on strategy
        if self.pooling_strategy == 'cls':
            # Use first token's embedding (position 0) - assumes BOS token is CLS
            if not has_bos:
                 logger.warning("CLS pooling requested but tokenizer doesn't seem to have a BOS token. Using first token.")
            embedding = token_embeddings[:, 0, :].cpu().numpy()[0]
        else:  # 'mean' pooling
            # Mean pool over the sequence length dimension (excluding special tokens)
            embedding = token_embeddings_no_special.mean(dim=1).cpu().numpy()[0]

        return embedding

    def batch_encode(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode multiple sequences in batches using ESM via Transformers""" # Updated docstring
        all_embeddings = []
        logger.info(f"Encoding {len(sequences)} sequences in batches of {batch_size} using Transformers/ESM...") # Updated log message

        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding (ESM - Transformers)"): # Updated tqdm desc
            batch_sequences = sequences[i:i + batch_size]
            # Add spaces between residues for ESM tokenizers
            batch_sequences_spaced = [" ".join(list(seq)) for seq in batch_sequences]

            # Tokenize batch with padding
            inputs = self.tokenizer(batch_sequences_spaced, return_tensors="pt", padding=True, truncation=True, max_length=1024) # Add truncation
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            attention_mask = inputs['attention_mask']

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_dim)

            # Remove BOS/EOS tokens if they exist before pooling
            has_bos = self.tokenizer.bos_token_id is not None
            has_eos = self.tokenizer.eos_token_id is not None
            start_idx = 1 if has_bos else 0
            end_idx = -1 if has_eos else None # This might be tricky with padding, use attention mask instead for mean

            if self.pooling_strategy == 'cls':
                 if i == 0 and not has_bos: logger.warning("CLS pooling requested but tokenizer doesn't seem to have a BOS token. Using first token.")
                 batch_embeddings = token_embeddings[:, 0, :].cpu().numpy()
            else: # 'mean' pooling
                 # Mask out padding tokens AND potentially BOS/EOS before averaging
                 # Create mask for actual sequence tokens (excluding padding AND special tokens)
                 sequence_mask = attention_mask.clone()
                 if has_bos:
                     sequence_mask[:, 0] = 0 # Mask out BOS
                 if has_eos:
                     # Mask out EOS - find first padding token (0) for each sequence
                     eos_indices = (attention_mask.sum(dim=1) - 1).long() # Index of last non-padding token
                     sequence_mask[torch.arange(attention_mask.size(0)), eos_indices] = 0 # Mask out EOS

                 # Expand mask for broadcasting: (batch, seq_len, 1)
                 sequence_mask_expanded = sequence_mask.unsqueeze(-1).float()
                 # Sum embeddings where mask is 1
                 sum_embeddings = (token_embeddings * sequence_mask_expanded).sum(dim=1)
                 # Count tokens where mask is 1 (avoid division by zero)
                 num_tokens = sequence_mask_expanded.sum(dim=1).clamp(min=1e-9)
                 # Calculate mean
                 batch_embeddings = (sum_embeddings / num_tokens).cpu().numpy()


            all_embeddings.append(batch_embeddings)

        # Check if any embeddings were generated before stacking
        if not all_embeddings:
            logger.warning("Batch encode generated no embeddings.")
            return np.array([]) # Return empty array

        return np.vstack(all_embeddings)

    # Override batch_encode_alleles (no changes needed here, uses self.batch_encode)
    def batch_encode_alleles(self, alleles: List[str], batch_size: int = 8, force: bool = False) -> Dict[str, np.ndarray]: # Added force parameter
        """Encode multiple alleles in batch using the ESM batch encoder""" # Updated docstring
        results = {}
        alleles_to_encode = []
        sequences_to_encode = []
        missing_sequences = []

        for allele in alleles:
            std_allele = self._standardize_allele(allele)
            # Skip cache check if force is True
            if not force and std_allele in self.embeddings:
                results[allele] = self.embeddings[std_allele]
            else:
                sequence = self.get_sequence(std_allele)
                if sequence:
                    alleles_to_encode.append(std_allele)
                    sequences_to_encode.append(sequence)
                else:
                    logger.warning(f"No sequence found for allele {allele} (standardized: {std_allele})")
                    missing_sequences.append(allele)

        if sequences_to_encode:
            logger.info(f"Batch encoding {len(sequences_to_encode)} sequences with ESM...") # Updated log message
            # Removed broad try...except to let specific errors propagate
            encoded_embeddings = self.batch_encode(sequences_to_encode, batch_size=batch_size)

            # Check for valid embeddings array immediately after the call
            if not isinstance(encoded_embeddings, np.ndarray):
                logger.error(f"Batch encoding returned non-array result: {type(encoded_embeddings)}")
                missing_sequences.extend(alleles_to_encode) # Mark all as missing
            elif encoded_embeddings.shape[0] != len(alleles_to_encode):
                 logger.error(f"Batch encoding returned wrong number of embeddings. Expected {len(alleles_to_encode)}, got {encoded_embeddings.shape[0]}")
                 # Decide how to handle partial failure - for now, mark all as missing
                 missing_sequences.extend(alleles_to_encode)
            else:
                # Proceed only if embeddings look valid
                logger.info(f"Assigning {encoded_embeddings.shape[0]} generated embeddings.")
                for i, allele_std_name in enumerate(alleles_to_encode):
                    # This should now be safe if the checks above passed
                    embedding_vector = encoded_embeddings[i]
                    original_alleles = [a for a in alleles if self._standardize_allele(a) == allele_std_name]
                    for original_allele in original_alleles:
                         results[original_allele] = embedding_vector
                    self.embeddings[allele_std_name] = embedding_vector

                # Save cache only if encoding was successful
                self._save_embedding_cache()

        if missing_sequences:
             unique_missing = sorted(list(set(missing_sequences)))
             logger.warning(f"Failed to encode or find sequence for {len(unique_missing)} alleles: {unique_missing}")

        return results
