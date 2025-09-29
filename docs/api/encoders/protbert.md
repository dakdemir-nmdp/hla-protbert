# ProtBERTEncoder

The `ProtBERTEncoder` class implements a specific encoder for HLA alleles using the ProtBERT transformer model.

**Module:** `src.models.protbert`

## Overview

The `ProtBERTEncoder` class extends the base `HLAEncoder` class and provides specialized functionality for encoding HLA sequences using ProtBERT, a transformer-based protein language model. It supports:
- Different token pooling strategies
- Optional peptide binding region extraction
- Batch processing for efficient encoding
- Fine-tuning capabilities for specific tasks

## Class Constructor

```python
ProtBERTEncoder(
    sequence_file: Union[str, Path], 
    cache_dir: Union[str, Path] = "./data/embeddings",
    model_name: str = "Rostlab/prot_bert",
    locus: Optional[str] = None,
    device: Optional[str] = None,
    pooling_strategy: str = "mean",
    use_peptide_binding_region: bool = True,
    verify_ssl: bool = True
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence_file` | `Union[str, Path]` | Path to pickle file with HLA sequences |
| `cache_dir` | `Union[str, Path]` | Directory to cache embeddings (default: "./data/embeddings") |
| `model_name` | `str` | Hugging Face model name or path to local model (default: "Rostlab/prot_bert") |
| `locus` | `Optional[str]` | HLA locus to encode (e.g., 'A', 'B', 'DRB1') |
| `device` | `Optional[str]` | Device to run model on ('cpu', 'cuda', or None for auto-detection) |
| `pooling_strategy` | `str` | How to pool token embeddings ('mean', 'cls', or 'attention') (default: "mean") |
| `use_peptide_binding_region` | `bool` | Whether to extract peptide binding region before encoding (default: True) |
| `verify_ssl` | `bool` | Whether to verify SSL certificates when downloading models (default: True) |

## Methods

### _encode_sequence

```python
_encode_sequence(sequence: str) -> np.ndarray
```

Encode a protein sequence using ProtBERT.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | `str` | Protein sequence |

#### Returns

`np.ndarray`: Embedding vector

### batch_encode

```python
batch_encode(sequences: List[str], batch_size: int = 8) -> np.ndarray
```

Encode multiple sequences in batches.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | List of sequences to encode |
| `batch_size` | `int` | Batch size for encoding (default: 8) |

#### Returns

`np.ndarray`: Array of embeddings, shape (len(sequences), embedding_dim)

### encode_alleles_for_locus

```python
encode_alleles_for_locus(locus: Optional[str] = None) -> Dict[str, np.ndarray]
```

Encode all alleles for a specified locus.

If locus is None, uses the encoder's locus if specified.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `locus` | `Optional[str]` | HLA locus to encode alleles for |

#### Returns

`Dict[str, np.ndarray]`: Dict mapping allele names to embeddings

### fine_tune

```python
fine_tune(
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
)
```

Fine-tune ProtBERT on HLA data.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_data` | `List[str]` | List of protein sequences for training |
| `labels` | `List[Union[int, float]]` | List of labels for training sequences |
| `validation_data` | `Optional[List[str]]` | Optional list of sequences for validation |
| `validation_labels` | `Optional[List[Union[int, float]]]` | Optional list of labels for validation |
| `output_dir` | `str` | Directory to save fine-tuned model (default: "./models/fine_tuned") |
| `epochs` | `int` | Number of training epochs (default: 3) |
| `batch_size` | `int` | Training batch size (default: 8) |
| `learning_rate` | `float` | Learning rate (default: 5e-5) |
| `task_type` | `str` | 'classification' or 'regression' (default: "classification") |
| `num_labels` | `int` | Number of labels for classification (ignored for regression) (default: 2) |

#### Returns

`Dict`: Dict with training metrics

## Protected Methods

These methods are intended for internal use by the `ProtBERTEncoder` class:

### _load_model

```python
_load_model()
```

Load ProtBERT model and tokenizer.

## Related Classes

### HLADataset

The `HLADataset` class is a PyTorch Dataset for HLA sequences.

```python
HLADataset(sequences, labels, tokenizer, max_length=512)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | List of protein sequences |
| `labels` | `List[Union[int, float]]` | List of labels (can be None for inference) |
| `tokenizer` | `BertTokenizer` | BERT tokenizer |
| `max_length` | `int` | Maximum sequence length (default: 512) |

## Example Usage

```python
from pathlib import Path
from src.models.protbert import ProtBERTEncoder

# Initialize encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings",
    locus="A",
    pooling_strategy="mean",
    use_peptide_binding_region=True
)

# Get embedding for a specific allele
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")

# Batch encode multiple alleles
alleles = ["A*01:01", "A*02:01", "A*03:01"]
embeddings = encoder.batch_encode_alleles(alleles)
for allele, emb in embeddings.items():
    print(f"Encoded {allele}: shape {emb.shape}")

# Encode all alleles for a specific locus
locus_embeddings = encoder.encode_alleles_for_locus("B")
print(f"Encoded {len(locus_embeddings)} alleles for locus B")

# Fine-tune for a classification task
train_seqs = [encoder.get_sequence(a) for a in ["A*01:01", "A*02:01", "A*03:01"]]
train_labels = [0, 1, 0]  # Binary classification example
encoder.fine_tune(
    train_data=train_seqs,
    labels=train_labels,
    epochs=1,
    output_dir="./models/custom_hla",
    task_type="classification"
)
```

## Notes

- The ProtBERT model requires the `transformers` library to be installed.
- When working with protein sequences, spaces are added between amino acids for tokenization.
- The encoder supports different pooling strategies to convert token-level embeddings to a sequence-level embedding:
  - `mean`: Average pooling across tokens (default)
  - `cls`: Use the [CLS] token embedding
  - `attention`: Use attention-weighted pooling
- Fine-tuning capabilities allow adapting the model for specific downstream tasks like classification or regression.
