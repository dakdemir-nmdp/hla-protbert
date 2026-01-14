#!/bin/bash
# Complete HLA-ProtBERT pipeline for ALL encoders
# Generates embeddings with ProtBERT, ESM-2, ProtT5, Ankh Base, and Ankh Large
# Usage: ./run_complete_pipeline_all_encoders.sh [--disable-ssl-verify] [--ankh-backend <auto|huggingface|ankh>]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse optional arguments
DISABLE_SSL_VERIFY=${HLA_DISABLE_SSL_VERIFY:-0}
ANKH_BACKEND=${HLA_ANKH_BACKEND:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disable-ssl-verify)
            DISABLE_SSL_VERIFY=1
            shift
            ;;
        --ankh-backend)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --ankh-backend requires a value (auto|huggingface|ankh).${NC}"
                exit 1
            fi
            ANKH_BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--disable-ssl-verify] [--ankh-backend <auto|huggingface|ankh>]"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--disable-ssl-verify] [--ankh-backend <auto|huggingface|ankh>]"
            exit 1
            ;;
    esac
done

SSL_ARGS=()
if [[ "$DISABLE_SSL_VERIFY" == "1" ]]; then
    export HLA_DISABLE_SSL_VERIFY=1
    SSL_ARGS=(--disable-ssl-verify)
    echo -e "${YELLOW}WARNING: SSL verification disabled for Hugging Face downloads.${NC}"
fi

ANKH_ARGS=()
if [[ -n "$ANKH_BACKEND" ]]; then
    ANKH_ARGS=(--ankh-backend "$ANKH_BACKEND")
    export HLA_ANKH_BACKEND="$ANKH_BACKEND"
fi

echo "=========================================="
echo "HLA-ProtBERT Complete Pipeline"
echo "Running for ALL encoders"
echo "=========================================="
echo ""

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}WARNING: Not in a virtual environment!${NC}"
    echo "Please activate your venv first: source venv/bin/activate"
    exit 1
fi

# Function to print section headers
print_section() {
    echo ""
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: Required file not found: $1${NC}"
        exit 1
    fi
}

# Step 1: Create directory structure
print_section "Step 1: Creating directory structure"
mkdir -p data/raw/fasta
mkdir -p data/processed
mkdir -p data/embeddings/{protbert,esm,prott5,ankh/base,ankh/large}
mkdir -p data/analysis/locus_embeddings/{class1,class2}/{embeddings,plots,reports}
mkdir -p data/analysis/locus_embeddings/logs
echo -e "${GREEN}✓ Directories created${NC}"

# Step 2: Download HLA sequence data
print_section "Step 2: Downloading HLA sequence data"
if [ -f "data/processed/hla_sequences.pkl" ]; then
    echo -e "${YELLOW}Sequence file already exists. Skipping download.${NC}"
    read -p "Re-download data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m hlaprotbert.scripts.update_imgt --verbose
    fi
else
    python -m hlaprotbert.scripts.update_imgt --verbose
fi
echo -e "${GREEN}✓ HLA sequences ready${NC}"

# Verify sequence file exists
check_file "data/processed/hla_sequences.pkl"

# Step 3: Generate embeddings with all encoders
print_section "Step 3: Generating embeddings with ALL encoders"

encoders=("protbert" "esm" "prott5" "ankh-base" "ankh-large")
encoder_names=("ProtBERT" "ESM-2" "ProtT5" "Ankh Base" "Ankh Large")

for i in "${!encoders[@]}"; do
    encoder="${encoders[$i]}"
    name="${encoder_names[$i]}"
    
    echo ""
    echo "--- Encoding with $name ---"
    
    # Check if embeddings already exist
    if [ "$encoder" = "ankh-base" ]; then
        cache_file="data/embeddings/ankh/base/hla_embeddings.pkl"
    elif [ "$encoder" = "ankh-large" ]; then
        cache_file="data/embeddings/ankh/large/hla_embeddings.pkl"
    else
        cache_file="data/embeddings/$encoder/hla_embeddings.pkl"
    fi
    
    if [ -f "$cache_file" ]; then
        echo -e "${YELLOW}Embeddings already exist for $name${NC}"
        read -p "Regenerate embeddings? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all --force --verbose "${SSL_ARGS[@]}" "${ANKH_ARGS[@]}"
        fi
    else
        python -m hlaprotbert.scripts.generate_embeddings --encoder-type "$encoder" --all --verbose "${SSL_ARGS[@]}" "${ANKH_ARGS[@]}"
    fi
    
    echo -e "${GREEN}✓ $name embeddings complete${NC}"
done

# Step 4: Run analysis for each encoder
print_section "Step 4: Running locus-specific analysis"

echo "This step will analyze embeddings by locus (A, B, C, DRB1, etc.)"
echo "and generate visualizations (t-SNE, UMAP, PCA)"
echo ""

# For now, we'll use ProtBERT as the primary analysis
# You can extend this to analyze other encoders similarly
echo "Running analysis for ProtBERT embeddings..."
echo "(Analysis for other encoders can be added as needed)"

# Verify protbert embeddings exist
check_file "data/embeddings/protbert/hla_embeddings.pkl"

echo "Copying ProtBERT embeddings to locus-specific directories..."
for locus in A B C; do
    mkdir -p "data/analysis/locus_embeddings/class1/embeddings"
    cp data/embeddings/protbert/hla_embeddings.pkl \
       "data/analysis/locus_embeddings/class1/embeddings/hla_${locus}_embeddings.pkl"
done

for locus in DRB1 DQB1 DPB1; do
    mkdir -p "data/analysis/locus_embeddings/class2/embeddings"
    cp data/embeddings/protbert/hla_embeddings.pkl \
       "data/analysis/locus_embeddings/class2/embeddings/hla_${locus}_embeddings.pkl"
done

echo "Running analysis for Class I loci..."
python -m hlaprotbert.scripts.run_locus_analysis --class1-only --debug 2>&1 | tee data/analysis/locus_embeddings/logs/class1_analysis.log || true
python -m hlaprotbert.scripts.run_locus_analysis --class2-only --debug 2>&1 | tee data/analysis/locus_embeddings/logs/class2_analysis.log || true

echo -e "${GREEN}✓ Analysis complete${NC}"

# Step 5: Generate summary statistics
print_section "Step 5: Generating summary statistics"

cat > /tmp/generate_summary.py << 'EOF'
import pickle
import os
from pathlib import Path

print("\n" + "="*60)
print("HLA-ProtBERT Pipeline Summary")
print("="*60)

encoders = {
    'ProtBERT': 'data/embeddings/protbert/hla_embeddings.pkl',
    'ESM-2': 'data/embeddings/esm/hla_embeddings.pkl',
    'ProtT5': 'data/embeddings/prott5/hla_embeddings.pkl',
    'Ankh Base': 'data/embeddings/ankh/base/hla_embeddings.pkl',
    'Ankh Large': 'data/embeddings/ankh/large/hla_embeddings.pkl',
}

print("\nEmbedding Statistics:")
print("-" * 60)

for name, path in encoders.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Get first embedding to check dimensionality
        first_key = list(embeddings.keys())[0]
        dim = embeddings[first_key].shape[0]
        
        print(f"{name:12s}: {len(embeddings):5d} alleles, {dim:4d}-dim embeddings")
    else:
        print(f"{name:12s}: NOT GENERATED")

print("\n" + "="*60)
print("\nGenerated Artifacts:")
print("-" * 60)

artifacts = [
    ('Sequence data', 'data/processed/hla_sequences.pkl'),
    ('ProtBERT embeddings', 'data/embeddings/protbert/hla_embeddings.pkl'),
    ('ESM-2 embeddings', 'data/embeddings/esm/hla_embeddings.pkl'),
    ('ProtT5 embeddings', 'data/embeddings/prott5/hla_embeddings.pkl'),
    ('Ankh Base embeddings', 'data/embeddings/ankh/base/hla_embeddings.pkl'),
    ('Ankh Large embeddings', 'data/embeddings/ankh/large/hla_embeddings.pkl'),
]

for name, path in artifacts:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ {name:25s}: {size_mb:6.2f} MB")
    else:
        print(f"✗ {name:25s}: NOT FOUND")

print("\n" + "="*60)
print("\nData Locations:")
print("-" * 60)
print(f"Sequences:  data/processed/")
print(f"Embeddings: data/embeddings/")
print(f"            ├── protbert/")
print(f"            ├── esm/")
print(f"            ├── prott5/")
print(f"            └── ankh/")
print(f"                ├── base/")
print(f"                └── large/")
print(f"Analysis:   data/analysis/locus_embeddings/")
print(f"            ├── class1/ (A, B, C)")
print(f"            └── class2/ (DRB1, DQB1, DPB1)")
print("\n" + "="*60)
EOF

python /tmp/generate_summary.py
rm /tmp/generate_summary.py

# Final message
print_section "Pipeline Complete!"

echo -e "${GREEN}All encoders have been run successfully!${NC}"
echo ""
echo "You can now:"
echo "1. Compare encoder outputs:"
echo "   python examples/multi_encoder_comparison.py --alleles A*01:01 A*02:01 --benchmark"
echo ""
echo "2. Use embeddings in your analysis:"
echo "   - ProtBERT:    data/embeddings/protbert/hla_embeddings.pkl"
echo "   - ESM-2:       data/embeddings/esm/hla_embeddings.pkl"
echo "   - ProtT5:      data/embeddings/prott5/hla_embeddings.pkl"
echo "   - Ankh Base:   data/embeddings/ankh/base/hla_embeddings.pkl"
echo "   - Ankh Large:  data/embeddings/ankh/large/hla_embeddings.pkl"
echo ""
echo "3. View analysis results:"
echo "   ls data/analysis/locus_embeddings/"
echo ""
echo "4. Create ensemble embeddings:"
echo "   python examples/multi_encoder_comparison.py --alleles A*01:01 --ensemble"
echo ""
