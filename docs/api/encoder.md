# HLAEncoder

The `HLAEncoder` class is the base class for HLA sequence encoders with common functionality.

**Module:** `src.models.encoder`

## Overview

The `HLAEncoder` class provides common functionality for different encoder implementations:
- Cache management for embeddings
- Fallback mechanisms for allele resolution
- Sequence retrieval and standardization

This is an abstract base class that should be subclassed by specific encoder implementations.

## Class Constructor

```python
HLAEncoder(
    sequence_file: Union[str, Path],
    cache_dir: Union[str, Path] = "./data/embeddings",
    locus: Optional[str] = None,
    verify_ssl: bool = False
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence_file` | `Union[str, Path]` | Path to pickle file with HLA sequences |
| `cache_dir` | `Union[str, Path]` | Directory to cache embeddings (default: "./data/embeddings") |
| `locus` | `Optional[str]` | HLA locus to encode (e.g., 'A', 'B', 'DRB1'). If provided, only alleles of this locus will be encoded |
| `verify_ssl` | `bool` | Whether to verify SSL certificates when downloading models (default: False) |

## Methods

### get_sequence

```python
get_sequence(allele: str) -> Optional[str]
```

Get sequence for an allele with fallbacks.

Will try multiple resolution methods if the allele is not directly found:
1. Direct lookup
2. Resolution to 2-field (if more fields provided)
3. ARD mapping (if pyard available)
4. Resolution to 1-field

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `allele` | `str` | HLA allele name |

#### Returns

`Optional[str]`: Protein sequence or None if not found

### get_embedding

```python
get_embedding(allele: str) -> np.ndarray
```

Get embedding for an allele.

If the embedding is cached, returns it directly. Otherwise, gets the sequence and encodes it.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `allele` | `str` | HLA allele name |

#### Returns

`np.ndarray`: Embedding vector

#### Raises

`ValueError`: If no sequence found for allele

### batch_encode_alleles

```python
batch_encode_alleles(alleles: List[str]) -> Dict[str, np.ndarray]
```

Encode multiple alleles in batch.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `alleles` | `List[str]` | List of HLA allele names |

#### Returns

`Dict[str, np.ndarray]`: Dict mapping allele names to embeddings

### find_similar_alleles

```python
find_similar_alleles(
    allele: str, 
    top_k: int = 5, 
    metric: str = 'cosine'
) -> List[Tuple[str, float]]
```

Find most similar alleles to the given allele.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `allele` | `str` | Query HLA allele |
| `top_k` | `int` | Number of similar alleles to return (default: 5) |
| `metric` | `str` | Similarity metric ('cosine', 'euclidean', 'manhattan') (default: 'cosine') |

#### Returns

`List[Tuple[str, float]]`: List of (allele_name, similarity_score) tuples

## Protected Methods

These methods are intended for internal use by the `HLAEncoder` class and its subclasses:

### _encode_sequence

```python
_encode_sequence(sequence: str) -> np.ndarray
```

Encode a protein sequence to a vector. This method should be implemented by subclasses.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | `str` | Protein sequence |

#### Returns

`np.ndarray`: Embedding vector

### _standardize_allele

```python
_standardize_allele(allele: str) -> str
```

Standardize allele name format.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `allele` | `str` | HLA allele name |

#### Returns

`str`: Standardized allele name

### _load_sequences

```python
_load_sequences() -> None
```

Load HLA sequences from file.

### _initialize_ard

```python
_initialize_ard() -> None
```

Initialize Antigen Recognition Domain (ARD) mapper if available.

### _load_embedding_cache

```python
_load_embedding_cache() -> Dict[str, np.ndarray]
```

Load cached embeddings from file.

#### Returns

`Dict[str, np.ndarray]`: Dict mapping allele names to embeddings

### _save_embedding_cache

```python
_save_embedding_cache() -> None
```

Save embeddings to cache file.

## Static Methods

### _cosine_similarity

```python
@staticmethod
_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
```

Compute cosine similarity between two vectors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a`, `b` | `np.ndarray` | Input vectors |

#### Returns

`float`: Cosine similarity (1 = identical, 0 = orthogonal)

### _euclidean_distance

```python
@staticmethod
_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float
```

Compute Euclidean distance between two vectors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a`, `b` | `np.ndarray` | Input vectors |

#### Returns

`float`: Euclidean distance

### _manhattan_distance

```python
@staticmethod
_manhattan_distance(a: np.ndarray, b: np.ndarray) -> float
```

Compute Manhattan distance between two vectors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a`, `b` | `np.ndarray` | Input vectors |

#### Returns

`float`: Manhattan distance

## Example Usage

```python
from pathlib import Path
from src.models.encoder import HLAEncoder

# Create a custom encoder subclass
class MyHLAEncoder(HLAEncoder):
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        # Implement your custom encoding logic here
        # For example, a simple one-hot encoding
        import numpy as np
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        encoding = np.zeros((len(sequence), len(amino_acids)))
        for i, aa in enumerate(sequence):
            if aa in amino_acids:
                encoding[i, amino_acids.index(aa)] = 1
        return encoding.flatten()
        
# Initialize your encoder
encoder = MyHLAEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings",
    locus="A"
)

# Get embedding for an allele
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")

# Find similar alleles
similar = encoder.find_similar_alleles("A*02:01", top_k=5)
for allele, score in similar:
    print(f"  {allele}: similarity={score:.4f}")
