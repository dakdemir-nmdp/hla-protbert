# HLA-ProtBERT

A comprehensive framework for encoding HLA alleles using ProtBERT.

## Features

- **Automated IMGT/HLA Database Management**: Easy downloading and updating of the latest HLA sequence data
- **ProtBERT Encoding**: Convert HLA alleles to high-dimensional protein embeddings
- **Locus-Specific Models**: Separate encoders for different HLA loci (A, B, C, DRB1, etc.)
- **Transplant Matching**: Advanced donor-recipient compatibility analysis
- **Efficient Caching**: Save time by caching both sequences and embeddings

## Installation

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages. You can use either `venv` (built into Python) or `conda`.

#### Using venv

```bash
# Clone the repository
git clone https://github.com/dakdemir-nmdp/hla-protbert.git
cd hla-protbert

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Using conda

```bash
# Clone the repository
git clone https://github.com/dakdemir-nmdp/hla-protbert.git
cd hla-protbert

# Create a conda environment
conda create -n hlaprotbert python=3.9
conda activate hlaprotbert
```


# use venv
```bash
# Clone the repository
git clone
cd hla-protbert

# Create a virtual environment
python -m venv venv
# activate the virtual environment
source venv/bin/activate
```

### Installing Dependencies

You can install the dependencies using pip with the provided requirements.txt file:

```bash
# Install required dependencies
pip install -r requirements.txt

# Optionally, install development dependencies
pip install pytest pytest-cov flake8 black isort
```

### Installing as a Package

Alternatively, you can install the package in development mode:

```bash
# Basic installation
pip install -e .

# Install with optional dependencies
pip install -e ".[analysis,nomenclature]"
```

### Requirements

The main dependencies are:
- Python 3.7+
- PyTorch
- Transformers
- NumPy
- Pandas
- scikit-learn
- BioPython
- Matplotlib
- Seaborn
- PyYAML
- tqdm
- Requests

Optional dependencies for advanced features:
- UMAP (for dimensionality reduction)
- ReportLab (for PDF report generation)
- PyARD (for HLA nomenclature resolution)

## Quick Start

### 1. Download IMGT/HLA Database

First, download and process the IMGT/HLA database:

```bash
python scripts/update_imgt.py
```

This will download the latest HLA sequences from the IMGT/HLA database and process them for use with the ProtBERT encoder.

### 2. Generate Embeddings

Next, generate embeddings for HLA alleles:

```bash
# Generate embeddings for all HLA-A alleles
python scripts/generate_embeddings.py --locus A --all

# Generate embeddings for specific alleles
python scripts/generate_embeddings.py --alleles A*01:01 A*02:01 B*07:02
```

### 3. Basic Usage

```python
from src.models.protbert import ProtBERTEncoder

# Initialize encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings"
)

# Get embedding for an allele
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")

# Find similar alleles
similar = encoder.find_similar_alleles("A*02:01", top_k=5)
for allele, score in similar:
    print(f"  {allele}: similarity={score:.4f}")
```

### 4. HLA Matching Analysis

```bash
python examples/donor_matching.py \
  --donor A*01:01 A*02:01 B*07:02 B*08:01 C*07:01 C*07:02 \
  --recipient A*01:01 A*24:02 B*07:02 B*15:01 C*03:04 C*07:01 \
  --report matching_report.pdf
```

## Command-Line Tools

### Update IMGT/HLA Database

```bash
python scripts/update_imgt.py [--force] [--verbose]
```

Options:
- `--force`: Force update even if database is current
- `--verbose`: Enable verbose logging

### Generate Embeddings

```bash
python scripts/generate_embeddings.py 
  [--locus LOCUS] 
  [--alleles ALLELES [ALLELES ...]] 
  [--all] 
  [--model MODEL] 
  [--device {cpu,cuda}]
```

Options:
- `--locus`: Generate embeddings for a specific locus only
- `--alleles`: List of specific alleles to encode
- `--all`: Generate embeddings for all known alleles
- `--model`: ProtBERT model name or path
- `--device`: Device to run model on (cpu or cuda)

## Examples

The `examples/` directory contains scripts demonstrating key functionality:

- `basic_encoding.py`: Basic HLA allele encoding
- `donor_matching.py`: Donor-recipient HLA matching analysis

## Directory Structure

```
hla-protbert/
├── data/
│   ├── raw/                  # IMGT/HLA database files
│   ├── processed/            # Preprocessed sequence data
│   └── embeddings/           # Cached embeddings
├── src/
│   ├── data/                 # Data handling modules
│   ├── models/               # Encoder and predictor models
│   ├── analysis/             # Analysis and visualization tools
│   └── utils/                # Utility functions
├── scripts/                  # Command-line scripts
├── examples/                 # Example usage scripts
├── setup.py                  # Package installation
└── README.md                 # This file
```

## Citation

If you use this framework in your research, please cite:

```
@software{hla_protbert,
  author = {Deniz Akdemir},
  title = {HLA-ProtBERT: A framework for encoding HLA alleles using ProtBERT},
  year = {2025},
  url = {https://github.com/dakdemir-nmdp/hla-protbert}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [IMGT/HLA Database](https://www.ebi.ac.uk/ipd/imgt/hla/) for providing HLA sequence data
- [ProtTrans](https://github.com/agemagician/ProtTrans) for protein language models
- [BioPython](https://biopython.org/) for sequence processing utilities
