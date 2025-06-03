# HLA Protein Encoders

A comprehensive framework for encoding HLA alleles using state-of-the-art protein language models like ProtBERT and ESM. This tool provides researchers with high-dimensional protein embeddings for HLA alleles, enabling advanced analysis in immunogenetics, transplantation matching, and immunotherapy research.

## Features

- **Automated IMGT/HLA Database Management**: Easy downloading and updating of the latest HLA sequence data
- **Multiple Encoders**: Supports ProtBERT and ESM for converting HLA alleles to high-dimensional protein embeddings
- **Locus-Specific Encoding**: Option to encode alleles for specific HLA loci (A, B, C, DRB1, etc.)
- **Transplant Matching**: Advanced donor-recipient compatibility analysis using protein embeddings
- **Efficient Caching**: Save time by caching both sequences and embeddings, organized by encoder type
- **Visualization**: Generate t-SNE, UMAP, and PCA plots for embedding analysis
- **Batch Processing**: Efficient batch encoding for large-scale analysis

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Pipeline](#complete-pipeline)
- [Usage Examples](#usage-examples)
- [Command-Line Tools](#command-line-tools)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM for basic usage, 16GB+ recommended for large-scale analysis
- CUDA-capable GPU (optional, for faster processing)

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
python scripts/update_imgt.py --verbose

# This will:
# - Download the latest HLA protein sequences
# - Parse and organize sequences by locus
# - Create a processed sequence file at data/processed/hla_sequences.pkl
```

### 2. Generate Embeddings

Generate protein embeddings for HLA alleles:

```bash
# Generate ProtBERT embeddings for all alleles
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose

# Generate embeddings for specific locus only
python scripts/generate_embeddings.py --encoder-type protbert --locus A --all

# Generate ESM embeddings
python scripts/generate_embeddings.py --encoder-type esm --all --verbose
```

### 3. Basic Python Usage

```python
from src.models.encoders import ProtBERTEncoder, ESMEncoder

# Initialize ProtBERT encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings/protbert"
)

# Get embedding for a single allele
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")  # (768,) for ProtBERT

# Find similar alleles
similar_alleles = encoder.find_similar_alleles("A*02:01", top_k=5)
for allele, similarity in similar_alleles:
    print(f"{allele}: {similarity:.4f}")

# Batch encode multiple alleles
alleles = ["A*01:01", "A*02:01", "B*07:02", "B*08:01"]
embeddings = encoder.batch_encode_alleles(alleles)
```

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
python scripts/update_imgt.py --verbose

# Step 3: Generate embeddings
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose

# Step 4: Create visualizations
python scripts/encode_sequences.py --encoder-type protbert --verbose

# Step 5: Run locus-specific analysis
python scripts/run_locus_analysis.py --class1-only --verbose
python scripts/run_locus_analysis.py --class2-only --verbose
```

## Usage Examples

### Example 1: Encoding Specific Alleles

```python
from src.models.encoders import ProtBERTEncoder

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
python scripts/generate_embeddings.py \
    --encoder-type protbert \
    --allele-file alleles.txt \
    --verbose
```

### Example 4: Visualization and Analysis

```python
# Run visualization script
python scripts/analyze_locus_embeddings.py \
    --locus A \
    --output-dir data/analysis/locus_embeddings/class1 \
    --verbose
```

## Command-Line Tools

### update_imgt.py - Download HLA Sequences

```bash
python scripts/update_imgt.py [options]

Options:
  --config FILE         Path to configuration file
  --data-dir DIR        Base data directory (default: data)
  --force               Force update even if database is current
  --verbose, -v         Enable verbose logging
```

### generate_embeddings.py - Generate Protein Embeddings

```bash
python scripts/generate_embeddings.py [options]

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
python scripts/encode_sequences.py [options]

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
python scripts/run_locus_analysis.py [options]

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
   python scripts/generate_embeddings.py --encoder-type protbert --all --batch-size 4
   
   # Use CPU instead of GPU
   python scripts/generate_embeddings.py --encoder-type protbert --all --device cpu
   ```

3. **IMGT/HLA Download Issues**
   ```bash
   # Check connection and retry
   python scripts/update_imgt.py --force --verbose
   
   # Manual download from https://www.ebi.ac.uk/ipd/imgt/hla/
   ```

4. **Missing Dependencies**
   ```bash
   # Reinstall all dependencies
   pip install --upgrade -r requirements.txt
   ```

For detailed examples and advanced usage, see [EXAMPLES.md](EXAMPLES.md).

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