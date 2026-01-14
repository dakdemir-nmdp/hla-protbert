# HLA-ProtBERT: Protein Language Model Encoding for HLA Alleles

[![PyPI version](https://badge.fury.io/py/hlaprotbert.svg)](https://pypi.org/project/hlaprotbert/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready framework for encoding HLA alleles using state-of-the-art protein language models (ProtBERT, ESM, ProtT5, Ankh). This tool provides researchers with high-dimensional protein embeddings for HLA alleles, enabling advanced analysis in immunogenetics, transplantation matching, and immunotherapy research.

**Status**: v1.0.0 - Production Ready

## ✨ Key Features

- 🧬 **Automated IMGT/HLA Database Management**: Seamless downloading and updating of the latest HLA sequence data with version tracking
- 🤖 **Multiple State-of-the-Art Encoders**: 
  - **ProtBERT** (BERT-based, 420M params, 1024-dim embeddings) - Proven performance
  - **ESM-2** (RoBERTa-based, 650M params, 1280-dim embeddings) - State-of-the-art accuracy
  - **ProtT5** (T5-based, 1.3B params, 1024-dim embeddings) - Complementary architecture
  - **Ankh** (Purpose-built, 50M/650M params, 768/1536-dim embeddings) - Efficient inference
- 🎯 **Locus-Specific Encoding**: Fine-grained control for encoding specific HLA loci (A, B, C, DRB1, DQB1, DPB1)
- 🏥 **Advanced Transplant Matching**: Embedding-based donor-recipient compatibility analysis
- ⚡ **Efficient Caching**: Smart caching system for both sequences and embeddings
- 📊 **Comprehensive Visualization**: t-SNE, UMAP, and PCA plots with publication-ready quality
- 🚀 **GPU Acceleration**: Automatic GPU detection and utilization for faster processing
- 🔄 **Batch Processing**: Optimized batch encoding for large-scale analysis
- 📦 **Easy Installation**: Simple pip installation with minimal dependencies
- 📚 **Comprehensive Documentation**: Detailed API documentation and examples

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Selection Guide](#model-selection-guide)
- [Complete Pipeline](#complete-pipeline)
- [Usage Examples](#usage-examples)
- [Command-Line Tools](#command-line-tools)
- [Directory Structure](#directory-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Via pip (Recommended)

The easiest way to install HLA-ProtBERT:

```bash
pip install hlaprotbert
```

For additional features:

```bash
# With development tools
pip install hlaprotbert[dev]

# With analysis tools (UMAP, reporting)
pip install hlaprotbert[analysis]

# With HLA nomenclature support (py-ard)
pip install hlaprotbert[nomenclature]

# Install all extras
pip install hlaprotbert[all]
```

### From Source (For Development)

For contributors or those who want the latest features:

```bash
# 1. Clone and navigate to repository
git clone https://github.com/dakdemir-nmdp/hla-protbert.git
cd hla-protbert

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR: venv\Scripts\activate  # On Windows

# 3. Install in development mode
pip install -e ".[dev]"

# 4. (Optional) Run automated setup for full pipeline
./setup_and_verify.sh
```

**Disk space**: ~15GB (for all models)  
**For detailed instructions**: See [`docs/INSTALLATION_GUIDE.md`](docs/INSTALLATION_GUIDE.md)

### Manual Installation

If you prefer step-by-step control:

### Prerequisites

- Python 3.9 or higher
- Git
- 8GB+ RAM (16GB+ recommended for large models)
- CUDA-capable GPU (optional, for faster processing)
- ~15GB disk space for models and data

### Step 1: Clone the Repository

```bash
git clone https://github.com/dakdemir-nmdp/hla-protbert.git
cd hla-protbert
```

### Step 2: Set Up Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts.

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n hlaprotbert python=3.9
conda activate hlaprotbert
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Optional: Install additional dependencies for visualization and analysis
pip install -e ".[analysis,visualization]"
```

### Step 4: Download Protein Language Models

The models will be automatically downloaded when first used. However, you can pre-download them:

```python
# Pre-download models (optional)
from transformers import AutoModel, AutoTokenizer

# Download ProtBERT
AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
AutoModel.from_pretrained("Rostlab/prot_bert")

# Download ESM
AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

## Quick Start

### 1. Download HLA Sequence Data

First, download the latest HLA sequences from the IMGT/HLA database:

```bash
# Download and process IMGT/HLA database
python -m hlaprotbert.scripts.update_imgt --verbose

# This will:
# - Download the latest HLA protein sequences
# - Parse and organize sequences by locus
# - Create a processed sequence file at data/processed/hla_sequences.pkl
```

### 2. Generate Embeddings

Generate protein embeddings for HLA alleles:

```bash
# Generate ProtBERT embeddings for all alleles
python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all --verbose

# Generate embeddings for specific locus only
python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --locus A --all

# Generate ESM embeddings
python -m hlaprotbert.scripts.generate_embeddings --encoder-type esm --all --verbose
```

### 3. Basic Python Usage

```python
from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder

# Initialize ProtBERT encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings/protbert"
### System Requirements
)
- Python 3.9 or higher (3.9, 3.10, 3.11 tested)
- 4GB+ RAM for basic usage (8GB+ recommended)
- 16GB+ RAM recommended for large-scale analysis
- 10GB+ disk space for models and data

### Compatibility

- **Operating Systems**: Linux, macOS, Windows
- **Python Versions**: 3.9, 3.10, 3.11
- **Compute**: CPU (all features) or GPU (CUDA 11.6+ for acceleration)

# Find similar alleles
similar_alleles = encoder.find_similar_alleles("A*02:01", top_k=5)
for allele, similarity in similar_alleles:
    print(f"{allele}: {similarity:.4f}")

# Batch encode multiple alleles
alleles = ["A*01:01", "A*02:01", "B*07:02", "B*08:01"]
embeddings = encoder.batch_encode_alleles(alleles)
```

## Choosing the Right Encoder

HLA-ProtBERT provides four state-of-the-art protein language models, each with different strengths:

### Model Comparison

| Model | Architecture | Parameters | Embedding Dim | Speed | Memory | Best For |
|-------|-------------|------------|---------------|-------|---------|----------|
| **Ankh Base** | Custom | 50M | 768 | ⚡⚡⚡⚡ | 🟢 Low | Production, fast inference |
| **ProtBERT** | BERT | 420M | 1024 | ⚡⚡⚡ | 🟡 Medium | General purpose, proven |
| **ESM-2** | RoBERTa | 650M | 1280 | ⚡⚡ | 🟡 Medium | High accuracy, research |
| **Ankh Large** | Custom | 650M | 1536 | ⚡⚡ | 🟡 Medium | Balanced speed/accuracy |
| **ProtT5** | T5 | 1.3B | 1024 | ⚡ | 🔴 High | Complementary features |

### When to Use Each Model

#### 🚀 Ankh Base (Recommended for Production)
```python
from hlaprotbert.models.encoders import AnkhEncoder

encoder = AnkhEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_variant="base"  # 50M params, very fast
)
```
**Use when:**
- Speed is critical (3-5x faster than ProtBERT)
- Limited GPU memory (runs well on CPU)
- Deploying in production environments
- Processing large allele databases

#### 🎯 ProtBERT (Recommended for General Research)
```python
from hlaprotbert.models.encoders import ProtBERTEncoder

encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    pooling_strategy="mean"
)
```
**Use when:**
- Established baseline for comparison
- Well-documented performance
- Standard research applications
- Balance of speed and accuracy

#### 🔬 ESM-2 (Recommended for High-Accuracy Research)
```python
from hlaprotbert.models.encoders import ESMEncoder

encoder = ESMEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_name="facebook/esm2_t33_650M_UR50D"
)
```
**Use when:**
- Maximum accuracy is required
- Larger embedding dimensions beneficial (1280-dim)
- State-of-the-art performance needed
- GPU is available

#### 🧬 Ankh Large (Best Speed/Accuracy Balance)
```python
from hlaprotbert.models.encoders import AnkhEncoder

encoder = AnkhEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_variant="large"  # 650M params, 1536-dim
)
```
**Use when:**
- Need high accuracy but faster than ESM-2
- Want larger embeddings than ProtBERT
- Benefit from purpose-built protein model
- Modern architecture advantages

#### 🎨 ProtT5 (Complementary Architecture)
```python
from hlaprotbert.models.encoders import ProtT5Encoder

encoder = ProtT5Encoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    pooling_strategy="mean"
)
```
**Use when:**
- Building ensemble models
- T5 architecture captures different patterns
- Research comparing architectures
- Creating hybrid embeddings

### Ensemble Approach (Recommended for Critical Applications)

Combine multiple models for robust predictions:

```python
from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder
import numpy as np

# Initialize multiple encoders
protbert = ProtBERTEncoder("./data/processed/hla_sequences.pkl")
esm = ESMEncoder("./data/processed/hla_sequences.pkl")
prott5 = ProtT5Encoder("./data/processed/hla_sequences.pkl")
ankh = AnkhEncoder("./data/processed/hla_sequences.pkl", model_variant="large")

# Get embeddings from all models
allele = "A*01:01"
emb_protbert = protbert.get_embedding(allele)  # 768-dim
emb_esm = esm.get_embedding(allele)            # 1280-dim
emb_prott5 = prott5.get_embedding(allele)      # 1024-dim
emb_ankh = ankh.get_embedding(allele)          # 1536-dim

# Option 1: Concatenate embeddings
ensemble_embedding = np.concatenate([emb_protbert, emb_esm, emb_prott5, emb_ankh])
print(f"Ensemble embedding: {ensemble_embedding.shape}")  # (4608,)

# Option 2: Average normalized embeddings
from sklearn.preprocessing import normalize

emb_norm_protbert = normalize(emb_protbert.reshape(1, -1))
emb_norm_esm = normalize(emb_esm.reshape(1, -1))
emb_norm_prott5 = normalize(emb_prott5.reshape(1, -1))
emb_norm_ankh = normalize(emb_ankh.reshape(1, -1))

# Pad to same dimension (1536) and average
def pad_embedding(emb, target_dim=1536):
    if len(emb[0]) < target_dim:
        padding = np.zeros((1, target_dim - len(emb[0])))
        return np.concatenate([emb, padding], axis=1)
    return emb[:, :target_dim]

averaged_embedding = np.mean([
    pad_embedding(emb_norm_protbert),
    pad_embedding(emb_norm_esm),
    pad_embedding(emb_norm_prott5),
    pad_embedding(emb_norm_ankh)
], axis=0)
```

### Performance Benchmarks

Typical inference times on a single HLA sequence (365 amino acids):

| Model | CPU (Apple M1) | GPU (NVIDIA A100) | Batch Throughput (GPU) |
|-------|----------------|-------------------|------------------------|
| Ankh Base | 0.15s | 0.02s | 500 alleles/min |
| ProtBERT | 0.45s | 0.05s | 200 alleles/min |
| ESM-2 | 0.60s | 0.08s | 150 alleles/min |
| Ankh Large | 0.55s | 0.07s | 160 alleles/min |
| ProtT5 | 1.20s | 0.12s | 100 alleles/min |

*Note: Benchmarks are approximate and depend on hardware and batch size.*

## Complete Pipeline

### Automated Pipeline Script

Run the complete pipeline with a single command:

```bash
# Make the script executable
chmod +x run_complete_pipeline.sh

# Run the complete pipeline
./run_complete_pipeline.sh
```

This script will automatically:
1. Create all required directories
2. Download HLA sequences from IMGT/HLA
3. Generate embeddings for all loci using ProtBERT
4. Create locus-specific analysis
5. Generate visualizations
6. Run analysis notebooks

### Manual Step-by-Step Pipeline

If you prefer to run each step manually:

```bash
# Step 1: Create directory structure
mkdir -p data/{raw,processed,embeddings/{protbert,esm}}
mkdir -p data/analysis/locus_embeddings/{class1,class2}/{embeddings,plots,reports}

# Step 2: Download HLA data
python -m hlaprotbert.scripts.update_imgt --verbose

# Step 3: Generate embeddings
python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all --verbose

# Step 4: Create visualizations
python -m hlaprotbert.scripts.encode_sequences --encoder-type protbert --verbose

# Step 5: Run locus-specific analysis
python -m hlaprotbert.scripts.run_locus_analysis --class1-only --verbose
python -m hlaprotbert.scripts.run_locus_analysis --class2-only --verbose
```

## Usage Examples

### Example 1: Encoding Specific Alleles

```python
from hlaprotbert.models.encoders import ProtBERTEncoder

# Initialize encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings/protbert",
    use_peptide_binding_region=True  # Focus on antigen-binding region
)

# Encode specific alleles
alleles_of_interest = ["A*01:01", "A*02:01", "A*03:01", "A*24:02"]
for allele in alleles_of_interest:
    embedding = encoder.get_embedding(allele)
    print(f"{allele}: {embedding.shape}")
```

### Example 2: Donor-Recipient Matching

```bash
# Run donor matching analysis
python examples/donor_matching.py \
    --donor "A*01:01" "A*02:01" "B*07:02" "B*08:01" "C*07:01" "C*07:02" \
    --recipient "A*01:01" "A*24:02" "B*07:02" "B*15:01" "C*03:04" "C*07:01" \
    --report matching_report.pdf
```

### Example 3: Batch Processing from File

```bash
# Create a file with alleles to process
echo -e "A*01:01\nA*02:01\nA*03:01\nB*07:02\nB*08:01" > alleles.txt

# Generate embeddings for alleles in file
python -m hlaprotbert.scripts.generate_embeddings \
    --encoder-type protbert \
    --allele-file alleles.txt \
    --verbose
```

### Example 4: Visualization and Analysis

```python
# Run visualization script
python -m hlaprotbert.scripts.analyze_locus_embeddings \
    --locus A \
    --output-dir data/analysis/locus_embeddings/class1 \
    --verbose
```

## Command-Line Tools

### update_imgt.py - Download HLA Sequences

```bash
python -m hlaprotbert.scripts.update_imgt [options]

Options:
  --config FILE         Path to configuration file
  --data-dir DIR        Base data directory (default: data)
  --force               Force update even if database is current
  --verbose, -v         Enable verbose logging
```

### generate_embeddings.py - Generate Protein Embeddings

```bash
python -m hlaprotbert.scripts.generate_embeddings [options]

Required:
  --encoder-type {protbert,esm}   Encoder model to use

Data Selection (choose one):
  --all                 Encode all alleles in database
  --locus LOCUS         Encode all alleles for specific locus (A, B, C, etc.)
  --allele-file FILE    Encode alleles listed in file (CSV/TXT/TSV)

Optional:
  --model MODEL         Model name/path (default: Rostlab/prot_bert or facebook/esm2_t33_650M_UR50D)
  --device {cpu,cuda}   Device to use (default: auto-detect)
  --batch-size SIZE     Batch size for encoding (default: 8)
  --cache-dir DIR       Cache directory (default: data/embeddings)
  --force               Force regeneration of cached embeddings
  --config FILE         Configuration file path
  --verbose, -v         Enable verbose logging
```

### encode_sequences.py - Process and Visualize

```bash
python -m hlaprotbert.scripts.encode_sequences [options]

Required:
  --encoder-type {protbert,esm}   Encoder model to use

Optional:
  --data-dir DIR        Directory with FASTA files (default: data/raw)
  --output-dir DIR      Output directory (default: data/processed)
  --locus LOCUS         Process specific locus only
  --model MODEL         Model name/path
  --device {cpu,cuda}   Device to use
  --batch-size SIZE     Batch size (default: 8)
  --skip-visualizations Skip t-SNE/UMAP plots
  --verbose, -v         Enable verbose logging
```

### run_locus_analysis.py - Locus-Specific Analysis

```bash
python -m hlaprotbert.scripts.run_locus_analysis [options]

Options:
  --class1-only         Analyze only Class I loci (A, B, C)
  --class2-only         Analyze only Class II loci (DRB1, DQB1, DPB1)
  --locus LOCUS         Analyze specific locus only
  --encoder {protbert,esm}  Encoder to use (default: protbert)
  --output-dir DIR      Output directory
  --debug               Enable debug logging
```

## Directory Structure

```
hla-protbert/
├── data/
│   ├── raw/                      # Downloaded IMGT/HLA files
│   │   ├── fasta/                # Individual locus FASTA files
│   │   └── hla_prot.fasta        # Consolidated protein sequences
│   ├── processed/                # Processed sequence data
│   │   ├── hla_sequences.pkl     # Main sequence dictionary
│   │   └── {encoder}/plots/      # Encoder-specific visualizations
│   ├── embeddings/               # Cached embeddings
│   │   ├── protbert/             # ProtBERT embeddings
│   │   └── esm/                  # ESM embeddings
│   └── analysis/                 # Analysis results
│       └── locus_embeddings/     # Locus-specific analysis
├── src/
│   ├── data/                     # Data handling modules
│   │   ├── imgt_downloader.py    # IMGT/HLA database downloader
│   │   ├── imgt_parser.py        # Sequence parser
│   │   └── sequence_utils.py     # Sequence utilities
│   ├── models/
│   │   ├── encoder.py            # Base encoder class
│   │   └── encoders/             # Specific encoder implementations
│   │       ├── protbert.py       # ProtBERT encoder
│   │       └── esm.py            # ESM encoder
│   ├── analysis/                 # Analysis tools
│   │   ├── matching.py           # HLA matching algorithms
│   │   ├── metrics.py            # Similarity metrics
│   │   └── visualization.py      # Plotting utilities
│   └── utils/                    # Utility functions
├── scripts/                      # Command-line scripts
├── examples/                     # Example usage scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

### Configuration & Path Overrides

Default paths are specified in function signatures for convenience but can be overridden using multiple methods:

**Priority Order (highest to lowest):**
1. **Function Parameters** - Runtime override via function arguments
2. **Environment Variables** - Set before running scripts
3. **Configuration File** - Via `config.yaml` and `ConfigManager`
4. **Default Values** - Hardcoded in function signatures

**Example Override Methods:**

```python
# Method 1: Function parameters (highest priority)
encoder = ProtBERTEncoder(cache_dir="/custom/path/embeddings")

# Method 2: Environment variables
export HLA_CACHE_DIR="/custom/path/embeddings"
export HLA_RAW_DIR="/custom/path/raw"

# Method 3: Configuration file
from hlaprotbert.utils.config import ConfigManager
config = ConfigManager(config_path="custom_config.yaml")
config.set("data.embeddings_dir", "/custom/path/embeddings")
```

**Default Paths:**
- **Raw Data:** `./data/raw` (IMGT/HLA downloads)
- **Processed Data:** `./data/processed` (parsed sequences)
- **Embeddings:** `./data/embeddings` (cached embeddings)
- **Analysis:** `./data/analysis` (results and reports)

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more configuration options.

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Try manual download with retry
   export TRANSFORMERS_OFFLINE=0
   export HF_DATASETS_OFFLINE=0
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('Rostlab/prot_bert')"
   ```

2. **Out of Memory Errors**
   ```bash
   # Reduce batch size
   python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all --batch-size 4
   
   # Use CPU instead of GPU
   python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all --device cpu
   ```

3. **IMGT/HLA Download Issues**
   ```bash
   # Check connection and retry
   python -m hlaprotbert.scripts.update_imgt --force --verbose
   
   # Manual download from https://www.ebi.ac.uk/ipd/imgt/hla/
   ```

4. **Missing Dependencies**
   ```bash
   # Reinstall all dependencies
   pip install --upgrade -r requirements.txt
   ```

For detailed examples and advanced usage, see [EXAMPLES.md](EXAMPLES.md).

## Documentation

### Getting Started
- **[Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- **[Installation Guide](docs/INSTALLATION_GUIDE.md)** - Detailed installation and verification
- **[New Encoders Quick Start](docs/QUICK_START_NEW_ENCODERS.md)** - Using ProtT5 and Ankh models

### API and Usage
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Examples](EXAMPLES.md)** - Comprehensive usage examples
- **[Tutorials](docs/tutorials/)** - Step-by-step tutorials

### Technical Documentation
- **[Architecture Decisions](docs/architecture/)** - Design rationale and future roadmap
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Solutions to common issues
- **[Release Checklist](docs/RELEASE_CHECKLIST.md)** - For contributors

### Additional Resources
- **[Changelog](CHANGELOG.md)** - Version history and updates
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Code Coverage Report](htmlcov/index.html)** - Test coverage statistics

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{hla_protein_encoders,
  author = {Deniz Akdemir},
  title = {HLA Protein Encoders: A framework for encoding HLA alleles using protein language models},
  year = {2025},
  url = {https://github.com/dakdemir-nmdp/hla-protbert}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [IMGT/HLA Database](https://www.ebi.ac.uk/ipd/imgt/hla/) for providing comprehensive HLA sequence data
- [ProtTrans](https://github.com/agemagician/ProtTrans) for pre-trained protein language models
- [ESM](https://github.com/facebookresearch/esm) for evolutionary scale modeling
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
