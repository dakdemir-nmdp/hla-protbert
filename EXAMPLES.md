# HLA-ProtBERT Data Pipeline Guide

This guide explains how to generate and organize the complete data structure for the HLA-ProtBERT project, including sequence downloads, embeddings generation, and visualization.

## Initial Setup

First, install all the required dependencies:

```bash
# Install base requirements
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install additional visualization dependencies
pip install umap-learn scikit-learn matplotlib seaborn pickle5 jupyter nbconvert
```

## Directory Structure Creation

Create the necessary directory structure:

```bash
# Create main directories
mkdir -p data/raw/fasta
mkdir -p data/processed
mkdir -p data/embeddings/protbert
mkdir -p data/embeddings/esm
mkdir -p data/analysis/locus_embeddings/class1/embeddings
mkdir -p data/analysis/locus_embeddings/class1/plots
mkdir -p data/analysis/locus_embeddings/class1/reports
mkdir -p data/analysis/locus_embeddings/class2/embeddings
mkdir -p data/analysis/locus_embeddings/class2/plots
mkdir -p data/analysis/locus_embeddings/class2/reports
mkdir -p data/analysis/locus_embeddings/logs
```

## Data Acquisition Pipeline

Download HLA sequence data from IMGT/HLA database:

```bash
# Download raw sequence data
python scripts/download_imgt_data.py --output-dir data/raw/fasta

# Process downloaded data
python scripts/imgt_parser.py --input-dir data/raw/fasta --output-file data/processed/hla_sequences.pkl
```

## Embeddings Generation Pipeline

Generate embeddings for all HLA loci using ProtBERT and/or ESM models:

```bash
# Generate ProtBERT embeddings for all loci
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose

# Generate ESM embeddings for all loci (optional)
python scripts/generate_embeddings.py --encoder-type esm --all --verbose

# Copy embeddings to locus-specific directories for analysis
python -c "
import os
import shutil
import glob

# Define source and destination directories
src_dir = 'data/embeddings/protbert'
class1_dir = 'data/analysis/locus_embeddings/class1/embeddings'
class2_dir = 'data/analysis/locus_embeddings/class2/embeddings'

# Create directories if they don't exist
os.makedirs(class1_dir, exist_ok=True)
os.makedirs(class2_dir, exist_ok=True)

# Copy the main embeddings file to each locus file
if os.path.exists(f'{src_dir}/hla_embeddings.pkl'):
    # Class I loci
    for locus in ['A', 'B', 'C']:
        dst_file = f'{class1_dir}/hla_{locus}_embeddings.pkl'
        shutil.copy(f'{src_dir}/hla_embeddings.pkl', dst_file)
        print(f'Created {dst_file}')
    
    # Class II loci
    for locus in ['DRB1', 'DQB1', 'DPB1']:
        dst_file = f'{class2_dir}/hla_{locus}_embeddings.pkl'
        shutil.copy(f'{src_dir}/hla_embeddings.pkl', dst_file)
        print(f'Created {dst_file}')
else:
    print('Source embeddings file not found')
"
```

## Visualization and Analysis Pipeline

Generate visualizations for all loci:

```bash
# Run locus analysis for Class I loci
python scripts/run_locus_analysis.py --class1-only --debug

# Run locus analysis for Class II loci
python scripts/run_locus_analysis.py --class2-only --debug

# Alternative: manually generate visualizations for each locus
python scripts/analyze_locus_embeddings.py --locus A --output-dir data/analysis/locus_embeddings/class1 --verbose
python scripts/analyze_locus_embeddings.py --locus B --output-dir data/analysis/locus_embeddings/class1 --verbose
python scripts/analyze_locus_embeddings.py --locus C --output-dir data/analysis/locus_embeddings/class1 --verbose
python scripts/analyze_locus_embeddings.py --locus DRB1 --output-dir data/analysis/locus_embeddings/class2 --verbose
python scripts/analyze_locus_embeddings.py --locus DQB1 --output-dir data/analysis/locus_embeddings/class2 --verbose
python scripts/analyze_locus_embeddings.py --locus DPB1 --output-dir data/analysis/locus_embeddings/class2 --verbose
```

## Running Jupyter Notebooks

After generating all data, you can run the notebooks:

```bash
# Run the B locus clustering notebook
jupyter nbconvert --to notebook --execute notebooks/b_locus_clustering.ipynb --output executed_b_locus_clustering.ipynb

# Run the HLA-ProtBERT demo notebook
jupyter nbconvert --to notebook --execute notebooks/hla_protbert_demo.ipynb --output executed_hla_protbert_demo.ipynb

# Run the locus embeddings analysis notebook
jupyter nbconvert --to notebook --execute notebooks/locus_embeddings_analysis.ipynb --output executed_locus_embeddings_analysis.ipynb
```

## Complete Data Pipeline in One Command

Here's a complete script to run the entire pipeline in one go:

```bash
#\!/bin/bash
# Full HLA-ProtBERT data generation pipeline

set -e  # Exit on error

echo "Creating directory structure..."
mkdir -p data/raw/fasta data/processed data/embeddings/{protbert,esm} \
  data/analysis/locus_embeddings/{class1,class2}/{embeddings,plots,reports} \
  data/analysis/locus_embeddings/logs

echo "Downloading HLA sequence data..."
python scripts/download_imgt_data.py --output-dir data/raw/fasta

echo "Processing downloaded data..."
python scripts/imgt_parser.py --input-dir data/raw/fasta --output-file data/processed/hla_sequences.pkl

echo "Generating ProtBERT embeddings for all loci..."
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose

echo "Copying embeddings to locus-specific directories..."
for locus in A B C; do
  cp data/embeddings/protbert/hla_embeddings.pkl data/analysis/locus_embeddings/class1/embeddings/hla_${locus}_embeddings.pkl
done

for locus in DRB1 DQB1 DPB1; do
  cp data/embeddings/protbert/hla_embeddings.pkl data/analysis/locus_embeddings/class2/embeddings/hla_${locus}_embeddings.pkl
done

echo "Running analysis for Class I loci..."
python scripts/run_locus_analysis.py --class1-only --debug

echo "Running analysis for Class II loci..."
python scripts/run_locus_analysis.py --class2-only --debug

echo "Pipeline complete\! You can now run the Jupyter notebooks."
```

## Troubleshooting

If you encounter issues during the pipeline execution:

1. **Missing dependencies**: Ensure all dependencies are installed using the setup commands above.

2. **Data download issues**: If IMGT/HLA data download fails, check your internet connection and try again, or download the files manually from [https://www.ebi.ac.uk/ipd/imgt/hla/](https://www.ebi.ac.uk/ipd/imgt/hla/).

3. **Embedding generation errors**: 
   - Make sure the processed sequences file exists at `data/processed/hla_sequences.pkl`
   - Check if you have sufficient disk space for the embeddings
   - Try reducing batch size if you encounter memory issues

4. **Visualization errors**:
   - Install visualization dependencies: `pip install matplotlib seaborn umap-learn scikit-learn`
   - If UMAP fails, try using only PCA and t-SNE by modifying the scripts

5. **Notebook execution errors**:
   - Make sure Jupyter is installed: `pip install jupyter nbconvert`
   - Check that all the data files referenced in the notebooks exist in the expected locations
   - Run notebooks manually in Jupyter if nbconvert fails
EOF < /dev/null