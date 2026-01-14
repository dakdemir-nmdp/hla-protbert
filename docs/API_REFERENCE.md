# API Reference

Complete API documentation for HLA-ProtBERT.

## Table of Contents

- [Encoders](#encoders)
  - [HLAEncoder (Base Class)](#hlaencoder-base-class)
  - [ProtBERTEncoder](#protbertencoder)
  - [ESMEncoder](#esmencoder)
- [Data Management](#data-management)
  - [IMGTDownloader](#imgtdownloader)
  - [IMGTParser](#imgtparser)
- [Analysis](#analysis)
  - [MatchingAnalyzer](#matchinganalyzer)
- [Configuration](#configuration)
  - [ConfigManager](#configmanager)

---

## Encoders

### HLAEncoder (Base Class)

Abstract base class for all HLA encoders.

```python
from hlaprotbert.models.encoder import HLAEncoder
```

#### Constructor

```python
HLAEncoder(
    sequence_file: Union[str, Path],
    cache_dir: Union[str, Path] = "./data/embeddings",
    locus: Optional[str] = None,
    verify_ssl: bool = False
)
```

**Parameters:**
- `sequence_file`: Path to pickle file containing HLA sequences
- `cache_dir`: Directory for caching embeddings (default: "./data/embeddings")
- `locus`: Optional HLA locus filter (e.g., "A", "B", "DRB1")
- `verify_ssl`: Whether to verify SSL certificates (default: False)

**Raises:**
- `TypeError`: If parameters have incorrect types
- `FileNotFoundError`: If sequence_file doesn't exist

#### Methods

##### get_embedding()

Get embedding vector for a single HLA allele.

```python
encoder.get_embedding(
    allele: str,
    force: bool = False
) -> np.ndarray
```

**Parameters:**
- `allele`: HLA allele identifier (e.g., "A*01:01")
- `force`: Regenerate embedding even if cached (default: False)

**Returns:**
- Numpy array of shape (embedding_dim,)

**Raises:**
- `TypeError`: If allele is not a string
- `ValueError`: If no sequence found for allele
- `RuntimeError`: If encoding fails

**Example:**
```python
embedding = encoder.get_embedding("A*01:01")
print(embedding.shape)  # (1024,) for ProtBERT
```

##### batch_encode_alleles()

Encode multiple HLA alleles efficiently.

```python
encoder.batch_encode_alleles(
    alleles: List[str],
    batch_size: int = 8,
    force: bool = False
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `alleles`: List of HLA allele identifiers
- `batch_size`: Number of sequences to process simultaneously (default: 8)
- `force`: Regenerate embeddings even if cached (default: False)

**Returns:**
- Dictionary mapping allele identifiers to embedding vectors

**Raises:**
- `TypeError`: If alleles is not a list or contains non-strings
- `ValueError`: If batch_size < 1

**Example:**
```python
alleles = ["A*01:01", "A*02:01", "B*07:02"]
embeddings = encoder.batch_encode_alleles(alleles, batch_size=16)
print(len(embeddings))  # 3
```

##### find_similar_alleles()

Find most similar alleles based on embedding similarity.

```python
encoder.find_similar_alleles(
    allele: str,
    top_k: int = 5,
    metric: str = 'cosine'
) -> List[Tuple[str, float]]
```

**Parameters:**
- `allele`: Query HLA allele identifier
- `top_k`: Number of most similar alleles to return (default: 5)
- `metric`: Distance metric - "cosine", "euclidean", or "manhattan" (default: "cosine")

**Returns:**
- List of (allele_name, similarity_score) tuples, sorted by similarity

**Raises:**
- `TypeError`: If parameters have incorrect types
- `ValueError`: If metric not supported or top_k < 1

**Example:**
```python
similar = encoder.find_similar_alleles("A*01:01", top_k=3)
for allele, score in similar:
    print(f"{allele}: {score:.3f}")
```

---

### ProtBERTEncoder

ProtBERT-based encoder for HLA alleles.

```python
from hlaprotbert.models.encoders.protbert import ProtBERTEncoder
```

#### Constructor

```python
ProtBERTEncoder(
    sequence_file: Union[str, Path],
    cache_dir: Union[str, Path] = "./data/embeddings/protbert",
    model_name: str = "Rostlab/prot_bert",
    locus: Optional[str] = None,
    device: Optional[str] = None,
    pooling_strategy: str = "mean",
    use_peptide_binding_region: bool = True,
    verify_ssl: bool = True
)
```

**Parameters:**
- `sequence_file`: Path to pickle file with HLA sequences
- `cache_dir`: Directory for caching embeddings (default: "./data/embeddings/protbert")
- `model_name`: Hugging Face model identifier (default: "Rostlab/prot_bert")
- `locus`: Optional HLA locus filter
- `device`: Device for computation - "cpu", "cuda", or None for auto-detection
- `pooling_strategy`: Token pooling method - "mean" or "cls" (default: "mean")
- `use_peptide_binding_region`: Focus on peptide-binding region (default: True)
- `verify_ssl`: Verify SSL certificates (default: True)

**Example:**
```python
encoder = ProtBERTEncoder(
    sequence_file="data/processed/hla_sequences.pkl",
    device="cuda",
    pooling_strategy="mean"
)

# Encode single allele
embedding = encoder.get_embedding("A*01:01")
print(embedding.shape)  # (1024,)

# Batch encoding
alleles = ["A*01:01", "A*02:01", "B*07:02"]
embeddings = encoder.batch_encode_alleles(alleles, batch_size=16)
```

---

### ESMEncoder

ESM-2 based encoder for HLA alleles.

```python
from hlaprotbert.models.encoders.esm import ESMEncoder
```

#### Constructor

```python
ESMEncoder(
    sequence_file: Union[str, Path],
    cache_dir: Union[str, Path] = "./data/embeddings/esm",
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    locus: Optional[str] = None,
    device: Optional[str] = None,
    pooling_strategy: str = "mean",
    verify_ssl: bool = True,
    hf_token: Optional[str] = None
)
```

**Parameters:**
- `sequence_file`: Path to pickle file with HLA sequences
- `cache_dir`: Directory for caching embeddings (default: "./data/embeddings/esm")
- `model_name`: Hugging Face model identifier (default: "facebook/esm2_t33_650M_UR50D")
- `locus`: Optional HLA locus filter
- `device`: Device for computation - "cpu", "cuda", or None for auto-detection
- `pooling_strategy`: Token pooling method - "mean" or "cls" (default: "mean")
- `verify_ssl`: Verify SSL certificates (default: True)
- `hf_token`: Hugging Face API token for authenticated downloads

**Example:**
```python
encoder = ESMEncoder(
    sequence_file="data/processed/hla_sequences.pkl",
    device="cuda",
    model_name="facebook/esm2_t33_650M_UR50D"
)

embedding = encoder.get_embedding("A*01:01")
print(embedding.shape)  # (1280,)
```

---

## Data Management

### IMGTDownloader

Downloads and manages IMGT/HLA database files.

```python
from hlaprotbert.data.imgt_downloader import IMGTDownloader
```

#### Constructor

```python
IMGTDownloader(
    data_dir: Union[str, Path] = './data/raw',
    use_github_first: bool = False,
    ftp_timeout: int = 30
)
```

**Parameters:**
- `data_dir`: Directory for storing downloaded data (default: "./data/raw")
- `use_github_first`: Try GitHub before FTP (default: False)
- `ftp_timeout`: FTP connection timeout in seconds (default: 30)

**Raises:**
- `TypeError`: If parameters have incorrect types
- `ValueError`: If ftp_timeout < 1

#### Methods

##### download_latest()

Download the latest IMGT/HLA database.

```python
downloader.download_latest(force: bool = False) -> bool
```

**Parameters:**
- `force`: Download even if current version exists (default: False)

**Returns:**
- True if download successful

**Raises:**
- `RuntimeError`: If all download sources fail
- `IOError`: If unable to write files

**Example:**
```python
downloader = IMGTDownloader(data_dir="./data/raw")
downloader.download_latest()
```

---

### IMGTParser

Parses IMGT/HLA FASTA files into structured sequences.

```python
from hlaprotbert.data.imgt_parser import IMGTParser
```

#### Constructor

```python
IMGTParser(
    imgt_dir: Union[str, Path] = './data/raw',
    output_dir: Union[str, Path] = './data/processed'
)
```

**Parameters:**
- `imgt_dir`: Directory containing raw IMGT/HLA files
- `output_dir`: Directory for processed output

**Raises:**
- `TypeError`: If parameters have incorrect types

#### Methods

##### parse_protein_sequences()

Parse protein sequences from FASTA files.

```python
parser.parse_protein_sequences() -> Dict[str, str]
```

**Returns:**
- Dictionary mapping allele identifiers to protein sequences

**Raises:**
- `FileNotFoundError`: If FASTA files/directories not found
- `IOError`: If unable to write output files

**Example:**
```python
parser = IMGTParser(
    imgt_dir="./data/raw",
    output_dir="./data/processed"
)
sequences = parser.parse_protein_sequences()
print(f"Parsed {len(sequences)} sequences")
```

---

## Analysis

### MatchingAnalyzer

Analyzes HLA matching between donors and recipients.

```python
from hlaprotbert.analysis.matching import MatchingAnalyzer
```

#### Constructor

```python
MatchingAnalyzer(
    encoder: HLAEncoder,
    loci: Optional[List[str]] = None,
    locus_weights: Optional[Dict[str, float]] = None,
    similarity_threshold: float = 0.9
)
```

**Parameters:**
- `encoder`: HLAEncoder instance for generating embeddings
- `loci`: List of HLA loci to consider (default: ["A", "B", "C", "DRB1", "DQB1", "DPB1"])
- `locus_weights`: Importance weights for each locus
- `similarity_threshold`: Threshold for functional similarity (default: 0.9)

**Raises:**
- `TypeError`: If encoder lacks required methods or loci not list
- `ValueError`: If similarity_threshold not in [0, 1]

**Example:**
```python
from hlaprotbert.models.encoders.protbert import ProtBERTEncoder

encoder = ProtBERTEncoder("data/processed/hla_sequences.pkl")
analyzer = MatchingAnalyzer(
    encoder=encoder,
    loci=["A", "B", "DRB1"],
    similarity_threshold=0.85
)
```

---

## Configuration

### ConfigManager

Manages application configuration with environment variable overrides.

```python
from hlaprotbert.utils.config import ConfigManager
```

#### Constructor

```python
ConfigManager(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Path to YAML or JSON configuration file

**Example:**
```python
config = ConfigManager(config_path="config.yml")
batch_size = config.get("model.batch_size")
```

---

## Type Hints Reference

All public APIs use comprehensive type hints:

```python
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
import numpy as np

# Common types
AlleleIdentifier = str
EmbeddingVector = np.ndarray  # Shape: (embedding_dim,)
EmbeddingDict = Dict[str, np.ndarray]
SimilarityResult = List[Tuple[str, float]]
```

## Error Handling

All methods use consistent error raising patterns:

- `TypeError`: For incorrect argument types
- `ValueError`: For invalid argument values
- `FileNotFoundError`: For missing required files
- `RuntimeError`: For runtime/operational errors
- `IOError`: For file I/O failures

Always catch specific exceptions rather than generic `Exception`.
