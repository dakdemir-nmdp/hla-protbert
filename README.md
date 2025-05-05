# HLA Protein Encoders

A comprehensive framework for encoding HLA alleles using various protein language models like ProtBERT and ESM.

## Features

- **Automated IMGT/HLA Database Management**: Easy downloading and updating of the latest HLA sequence data
- **Multiple Encoders**: Supports ProtBERT and ESM for converting HLA alleles to high-dimensional protein embeddings. Easily extensible for other models.
- **Locus-Specific Encoding**: Option to encode alleles for specific HLA loci (A, B, C, DRB1, etc.)
- **Transplant Matching**: Advanced donor-recipient compatibility analysis (using ProtBERT embeddings currently)
- **Efficient Caching**: Save time by caching both sequences and embeddings, organized by encoder type.

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
- transformers (for ESM models)

Optional dependencies for advanced features:
- UMAP (for dimensionality reduction)
- ReportLab (for PDF report generation)
- PyARD (for HLA nomenclature resolution)

## Complete Data Pipeline

For a streamlined process to generate all data, embeddings, and visualizations, we provide a complete pipeline script:

```bash
# Execute the complete pipeline script
./run_complete_pipeline.sh
```

This script will:
1. Create all required directories
2. Download and process HLA sequence data
3. Generate embeddings for all loci
4. Run locus-specific analysis for both Class I and Class II loci
5. Execute all notebooks to generate visualizations

For detailed instructions on each step of the pipeline, see [EXAMPLES.md](EXAMPLES.md).

## Quick Start

### 1. Download IMGT/HLA Database

First, download and process the IMGT/HLA database:

```bash
python scripts/update_imgt.py
```

This will download the latest HLA sequences from the IMGT/HLA database and process them for use with the ProtBERT encoder.

### 2. Generate Embeddings

Next, generate embeddings for HLA alleles using a specific encoder:

```bash
# Generate ProtBERT embeddings for all HLA-A alleles
python scripts/generate_embeddings.py --encoder-type protbert --locus A --all

# Generate ESM embeddings for specific alleles
python scripts/generate_embeddings.py --encoder-type esm --locus A --allele-file data/analysis/locus_embeddings/embeddings/A_alleles_esm.txt

# Generate ProtBERT embeddings for specific alleles listed directly
python scripts/generate_embeddings.py --encoder-type protbert --locus A --allele-file data/analysis/locus_embeddings/embeddings/specific_alleles.txt
```

### 3. Basic Usage

```python
from src.models.encoders import ProtBERTEncoder, ESMEncoder

# --- Using ProtBERT ---
# Initialize encoder (cache will be in ./data/embeddings/protbert/)
protbert_encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings" # Base dir, class handles subdir
)

# Get embedding for an allele
embedding_pb = protbert_encoder.get_embedding("A*01:01")
print(f"ProtBERT Embedding shape: {embedding_pb.shape}")

# Find similar alleles using ProtBERT embeddings
similar_pb = protbert_encoder.find_similar_alleles("A*02:01", top_k=5)
print("Similar (ProtBERT):")
for allele, score in similar_pb:
    print(f"  {allele}: similarity={score:.4f}")


# --- Using ESM ---
# Initialize encoder (cache will be in ./data/embeddings/esm/)
esm_encoder = ESMEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings" # Base dir, class handles subdir
)

# Get embedding for an allele
embedding_esm = esm_encoder.get_embedding("A*01:01")
print(f"\nESM Embedding shape: {embedding_esm.shape}")

# Find similar alleles using ESM embeddings
similar_esm = esm_encoder.find_similar_alleles("A*02:01", top_k=5)
print("\nSimilar (ESM):")
for allele, score in similar_esm:
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
  --encoder-type {protbert,esm}
  [--locus LOCUS]
  [--allele-file FILE]
  [--all]
  [--model MODEL]
  [--device {cpu,cuda}]
  [--batch-size SIZE]
  [--cache-dir DIR]
  [--force]
  [--verbose]
```

Options:
- `--encoder-type`: (Required) Choose the encoder model (`protbert` or `esm`).
- `--locus`: Generate embeddings for a specific locus only.
- `--allele-file`: Path to a file (CSV, TXT, TSV) containing alleles to encode.
- `--all`: Generate embeddings for all known alleles (from the sequence file).
- `--config`: Path to a configuration file.
- `--data-dir`: Base data directory.
- `--sequences`: Path to HLA sequences pickle file.
- `--model`: Model name or path for the selected encoder (e.g., `Rostlab/prot_bert` or `facebook/esm2_t33_650M_UR50D`). Defaults are provided.
- `--device`: Device to run model on (`cpu` or `cuda`). Auto-detects if not specified.
- `--batch-size`: Batch size for encoding (default: 8).
- `--cache-dir`: Base directory for caching embeddings (default: `data/embeddings`). Encoder-specific subdirectories (`protbert`, `esm`) will be created.
- `--force`: Force regeneration of existing cached embeddings.
- `--verbose`: Enable verbose logging.

### Encode Sequences and Visualize

```bash
python scripts/encode_sequences.py
  --encoder-type {protbert,esm}
  [--locus LOCUS]
  [--model MODEL]
  [--device {cpu,cuda}]
  [--batch-size SIZE]
  [--output-dir DIR]
  [--skip-visualizations]
  [--verbose]
```

Options:
- `--encoder-type`: (Required) Choose the encoder model (`protbert` or `esm`).
- `--locus`: Process only a specific locus. If omitted, processes all loci.
- `--model`: Model name or path for the selected encoder.
- `--device`: Device to run model on.
- `--batch-size`: Batch size for encoding (default: 8).
- `--output-dir`: Base directory for outputs (default: `data/processed`). The `hla_sequences.pkl` file is saved here. Encoder-specific subdirectories (`protbert/plots`, `esm/plots`) are created for visualizations.
- `--skip-visualizations`: Skip generating t-SNE and UMAP plots.
- `--verbose`: Enable verbose logging.

## Examples

The `examples/` directory contains scripts demonstrating key functionality:

- `basic_encoding.py`: Basic HLA allele encoding
- `donor_matching.py`: Donor-recipient HLA matching analysis

For a detailed guide on generating all data and visualizations from scratch, see the [EXAMPLES.md](EXAMPLES.md) file.

## Directory Structure

```
hla-protbert/
├── data/
│   ├── raw/                  # IMGT/HLA database files
│   ├── processed/            # Preprocessed sequence data (hla_sequences.pkl)
│   │   ├── protbert/         # ProtBERT specific outputs (e.g., plots)
│   │   └── esm/              # ESM specific outputs (e.g., plots)
│   ├── embeddings/           # Cached embeddings
│   │   ├── protbert/         # ProtBERT embeddings cache
│   │   └── esm/              # ESM embeddings cache
│   └── analysis/             # Analysis results
│       └── locus_embeddings/ # Locus-specific analysis
│           ├── class1/       # Class I loci (A, B, C)
│           │   ├── embeddings/
│           │   ├── plots/
│           │   └── reports/
│           └── class2/       # Class II loci (DRB1, DQB1, DPB1)
│               ├── embeddings/
│               ├── plots/
│               └── reports/
├── src/
│   ├── data/                 # Data handling modules (downloader, parser)
│   ├── models/
│   │   ├── encoders/         # Encoder implementations (protbert.py, esm.py)
│   │   └── encoder.py        # Base HLAEncoder class
│   ├── analysis/             # Analysis and visualization tools
│   └── utils/                # Utility functions (config, logging)
├── scripts/                  # Command-line scripts (generate_embeddings.py, etc.)
├── examples/                 # Example usage scripts
├── setup.py                  # Package installation
└── README.md                 # This file
```

## Citation

If you use this framework in your research, please cite:

```
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

- [IMGT/HLA Database](https://www.ebi.ac.uk/ipd/imgt/hla/) for providing HLA sequence data
- [ProtTrans](https://github.com/agemagician/ProtTrans) for protein language models
- [BioPython](https://biopython.org/) for sequence processing utilities
