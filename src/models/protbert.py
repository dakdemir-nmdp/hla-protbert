"""
ProtBERT HLA Encoder
-------------------
Implementation of HLA encoder using ProtBERT.
"""
import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

# Import base encoder
from .encoder import HLAEncoder

logger = logging.getLogger(__name__)

# Check for transformers library
try:
    from transformers import BertModel, BertTokenizer, BertForSequenceClassification
    from transformers import TrainingArguments, Trainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not installed; ProtBERT encoding not available")
    TRANSFORMERS_AVAILABLE = False

class HLADataset(torch.utils.data.Dataset):
    """PyTorch Dataset for HLA sequences"""
    
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        """Initialize dataset
        
        Args:
            sequences: List of protein sequences
            labels: List of labels (can be None for inference)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        # For protein sequences, we need spaces between amino acids
        spaced_sequence = " ".join(sequence)
        encoding = self.tokenizer(
            spaced_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Remove batch dimension added by tokenizer
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        
        # Add label if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

class ProtBERTEncoder(HLAEncoder):
    """ProtBERT-based encoder for HLA alleles"""
    
    def __init__(
        self, 
        sequence_file: Union[str, Path], 
        cache_dir: Union[str, Path] = "./data/embeddings",
        model_name: str = "Rostlab/prot_bert",
        locus: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        use_peptide_binding_region: bool = True,
        verify_ssl: bool = True
    ):
        """Initialize ProtBERT encoder
        
        Args:
            sequence_file: Path to pickle file with HLA sequences
            cache_dir: Directory to cache embeddings
            model_name: Hugging Face model name or path to local model
            locus: HLA locus to encode (e.g., 'A', 'B', 'DRB1')
            device: Device to run model on ('cpu', 'cuda', or None for auto-detection)
            pooling_strategy: How to pool token embeddings ('mean', 'cls', or 'attention')
            use_peptide_binding_region: Whether to extract peptide binding region before encoding
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not installed; cannot use ProtBERTEncoder")
            
        # Initialize base class
        super().__init__(sequence_file, cache_dir, locus, verify_ssl)
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.use_peptide_binding_region = use_peptide_binding_region
        
        # Set device
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load ProtBERT model and tokenizer"""
        logger.info(f"Loading ProtBERT model: {self.model_name}")
        
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name, 
                do_lower_case=False,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
            self.model = BertModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model on {self.device}")
        except Exception as e:
            logger.error(f"Error loading ProtBERT model: {e}")
            raise
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a protein sequence using ProtBERT
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Embedding vector
        """
        # Extract peptide binding region if requested
        if self.use_peptide_binding_region and self.locus:
            from ..data.sequence_utils import SequenceProcessor
            processor = SequenceProcessor()
            sequence = processor.extract_peptide_binding_region(sequence, self.locus)
        
        # Add spaces between amino acids for tokenization
        spaced_sequence = " ".join(sequence)
        
        # Tokenize
        inputs = self.tokenizer(
            spaced_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pool token embeddings based on strategy
        if self.pooling_strategy == 'cls':
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        elif self.pooling_strategy == 'attention':
            # Pool with attention weights
            hidden_states = outputs.last_hidden_state
            attention = torch.softmax(
                torch.matmul(hidden_states, hidden_states.transpose(1, 2)), dim=2
            )
            embedding = torch.matmul(attention, hidden_states).cpu().numpy()[0, 0, :]
        else:  # 'mean' pooling (default)
            # Mean across tokens, excluding padding
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            # Mask and compute mean
            embedding = ((hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / 
                       attention_mask.sum(dim=1, keepdim=True)).cpu().numpy()[0]
        
        return embedding
    
    def batch_encode(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode multiple sequences in batches
        
        Args:
            sequences: List of sequences to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings, shape (len(sequences), embedding_dim)
        """
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding"):
            batch = sequences[i:i + batch_size]
            
            # Add spaces between amino acids
            spaced_sequences = [" ".join(seq) for seq in batch]
            
            # Tokenize
            inputs = self.tokenizer(
                spaced_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pool token embeddings based on strategy
            if self.pooling_strategy == 'cls':
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.pooling_strategy == 'attention':
                # Pool with attention weights
                hidden_states = outputs.last_hidden_state
                attention = torch.softmax(
                    torch.matmul(hidden_states, hidden_states.transpose(1, 2)), dim=2
                )
                batch_embeddings = torch.matmul(attention, hidden_states)[:, 0, :].cpu().numpy()
            else:  # 'mean' pooling (default)
                # Mean across tokens, excluding padding
                attention_mask = inputs['attention_mask']
                hidden_states = outputs.last_hidden_state
                # Mask and compute mean
                batch_embeddings = ((hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / 
                           attention_mask.sum(dim=1, keepdim=True)).cpu().numpy()
                
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def encode_alleles_for_locus(self, locus: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Encode all alleles for a specified locus
        
        If locus is None, uses the encoder's locus if specified.
        
        Args:
            locus: HLA locus to encode alleles for
            
        Returns:
            Dict mapping allele names to embeddings
        """
        if locus is None:
            locus = self.locus
        
        if locus is None:
            logger.error("No locus specified for encoding")
            return {}
        
        # Get alleles for locus
        alleles = [
            allele for allele in self.sequences.keys()
            if allele.startswith(f"{locus}*")
        ]
        
        if not alleles:
            logger.warning(f"No alleles found for locus {locus}")
            return {}
            
        logger.info(f"Encoding {len(alleles)} alleles for locus {locus}")
        
        # Encode alleles
        return self.batch_encode_alleles(alleles)
    
    def fine_tune(
        self, 
        train_data: List[str], 
        labels: List[Union[int, float]], 
        validation_data: Optional[List[str]] = None,
        validation_labels: Optional[List[Union[int, float]]] = None,
        output_dir: str = "./models/fine_tuned",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        task_type: str = "classification",
        num_labels: int = 2
    ):
        """Fine-tune ProtBERT on HLA data
        
        Args:
            train_data: List of protein sequences for training
            labels: List of labels for training sequences
            validation_data: Optional list of sequences for validation
            validation_labels: Optional list of labels for validation
            output_dir: Directory to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            task_type: 'classification' or 'regression'
            num_labels: Number of labels for classification (ignored for regression)
        
        Returns:
            Dict with training metrics
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine model type
        problem_type = "regression" if task_type == "regression" else "single_label_classification"
        
        # Load model for sequence classification
        logger.info(f"Loading model for fine-tuning ({task_type} task)")
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1 if task_type == "regression" else num_labels,
            problem_type=problem_type
        )
        model.to(self.device)
        
        # Create datasets
        train_dataset = HLADataset(train_data, labels, self.tokenizer)
        
        if validation_data and validation_labels:
            valid_dataset = HLADataset(validation_data, validation_labels, self.tokenizer)
        else:
            valid_dataset = None
        
        # Setup training parameters
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            save_strategy="epoch",
            evaluation_strategy="epoch" if valid_dataset else "no",
            load_best_model_at_end=True if valid_dataset else False,
        )
        
        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset
        )
        
        # Train model
        logger.info("Starting fine-tuning")
        trainer.train()
        
        # Evaluate
        if valid_dataset:
            logger.info("Evaluating fine-tuned model")
            metrics = trainer.evaluate()
            logger.info(f"Evaluation metrics: {metrics}")
        else:
            metrics = {"train_loss": trainer.state.log_history[-1]["loss"]}
        
        # Save fine-tuned model
        logger.info(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Update our model to use the fine-tuned version
        logger.info("Updating encoder to use fine-tuned model")
        self.model_name = str(output_dir)  # Update model path
        self._load_model()  # Reload model
        
        # Clear embedding cache since the model has changed
        self.embeddings = {}
        self._save_embedding_cache()
        
        return metrics
