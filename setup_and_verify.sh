#!/bin/bash
# HLA-ProtBERT Setup and Installation Script
# This script installs all dependencies and verifies model availability

set -e  # Exit on error

echo "=========================================="
echo "HLA-ProtBERT Setup and Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo -e "${YELLOW}WARNING: Not running in a virtual environment!${NC}"
        echo "It's recommended to activate your venv first:"
        echo "  source venv/bin/activate"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
    fi
}

# Install Python dependencies
install_dependencies() {
    echo ""
    echo "Step 1: Installing Python dependencies..."
    echo "=========================================="
    
    # Upgrade pip first
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    
    # Install package in development mode
    echo "Installing hla-protbert in development mode..."
    pip install -e .
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Verify Python packages
verify_packages() {
    echo ""
    echo "Step 2: Verifying installed packages..."
    echo "=========================================="
    
    packages=("torch" "transformers" "numpy" "pandas" "scikit-learn" "sentencepiece" "ankh")
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            version=$(python -c "import $package; print($package.__version__)" 2>/dev/null)
            echo -e "${GREEN}✓${NC} $package ($version)"
        else
            echo -e "${RED}✗${NC} $package - NOT INSTALLED"
            exit 1
        fi
    done
    
    # Check transformers version
    transformers_version=$(python -c "import transformers; print(transformers.__version__)")
    if python -c "from packaging import version; import transformers; exit(0 if version.parse(transformers.__version__) >= version.parse('4.30.0') else 1)"; then
        echo -e "${GREEN}✓ transformers version $transformers_version is sufficient (>= 4.30.0)${NC}"
    else
        echo -e "${RED}✗ transformers version $transformers_version is too old (need >= 4.30.0)${NC}"
        exit 1
    fi
}

# Test model downloads
test_model_downloads() {
    echo ""
    echo "Step 3: Testing model downloads..."
    echo "=========================================="
    echo "This will download models (may take several minutes)..."
    echo ""
    
    # Create test script
cat > /tmp/test_encoders.py << 'EOF'
import sys
import logging
logging.basicConfig(level=logging.INFO)


def try_native_ankh_loader(loader_name: str, label: str) -> bool:
    try:
        import ankh
    except Exception as pkg_err:  # pragma: no cover - defensive
        print(f"✗ {label} fallback via 'ankh' package unavailable: {pkg_err}")
        return False

    try:
        loader = getattr(ankh, loader_name)
    except AttributeError:
        print(f"✗ {label} fallback loader '{loader_name}' not found in 'ankh' package")
        return False

    model, tokenizer = loader()
    model.eval()
    print(f"✓ {label} loaded via 'ankh' package")
    return True

print("\n--- Testing ProtBERT model download ---")
try:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    print("✓ ProtBERT model downloaded successfully")
except Exception as e:
    print(f"✗ ProtBERT failed: {e}")
    sys.exit(1)

print("\n--- Testing ESM-2 model download ---")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    print("✓ ESM-2 model downloaded successfully")
except Exception as e:
    print(f"✗ ESM-2 failed: {e}")
    sys.exit(1)

print("\n--- Testing ProtT5 model download ---")
try:
    from transformers import T5Tokenizer, T5EncoderModel
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", legacy=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    print("✓ ProtT5 model downloaded successfully")
except Exception as e:
    print(f"✗ ProtT5 failed: {e}")
    sys.exit(1)

print("\n--- Testing Ankh Base model download ---")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base", trust_remote_code=True)
    model = AutoModel.from_pretrained("ElnaggarLab/ankh-base", trust_remote_code=True)
    print("✓ Ankh Base model downloaded successfully (Hugging Face)")
except Exception as e:
    print(f"! Hugging Face Ankh Base download failed: {e}")
    if not try_native_ankh_loader("load_ankh_base", "Ankh Base"):
        sys.exit(1)

print("\n--- Testing Ankh Large model download ---")
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large", trust_remote_code=True)
    model = AutoModel.from_pretrained("ElnaggarLab/ankh-large", trust_remote_code=True)
    print("✓ Ankh Large model downloaded successfully (Hugging Face)")
except Exception as e:
    print(f"! Hugging Face Ankh Large download failed: {e}")
    if not try_native_ankh_loader("load_ankh_large", "Ankh Large"):
        sys.exit(1)

print("\n" + "="*50)
print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("="*50)
EOF
    
    python /tmp/test_encoders.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All models verified${NC}"
    else
        echo -e "${RED}✗ Model download failed${NC}"
        exit 1
    fi
    
    rm /tmp/test_encoders.py
}

# Test encoder functionality
test_encoder_functionality() {
    echo ""
    echo "Step 4: Testing encoder functionality..."
    echo "=========================================="
    
    # Create a minimal test
    cat > /tmp/test_encoder_function.py << 'EOF'
import sys
import os
import tempfile
import pickle
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create minimal test data
test_sequences = {
    'A*01:01': 'MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDRNTRNVKAQSQTDRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLFGAVIAGAVVAAVMWRRKSSDRKGGSYSQAAVSHDSAQGSDVSLTACKV',
}

# Create temp file
with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
    temp_file = f.name
    pickle.dump(test_sequences, f)

try:
    from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder
    
    print("\n--- Testing ProtBERT encoding ---")
    encoder = ProtBERTEncoder(temp_file, device='cpu')
    embedding = encoder.get_embedding('A*01:01')
    assert embedding.shape == (768,), f"Wrong shape: {embedding.shape}"
    print(f"✓ ProtBERT: {embedding.shape}")
    
    print("\n--- Testing ESM-2 encoding ---")
    encoder = ESMEncoder(temp_file, device='cpu')
    embedding = encoder.get_embedding('A*01:01')
    assert embedding.shape == (1280,), f"Wrong shape: {embedding.shape}"
    print(f"✓ ESM-2: {embedding.shape}")
    
    print("\n--- Testing ProtT5 encoding ---")
    encoder = ProtT5Encoder(temp_file, device='cpu')
    embedding = encoder.get_embedding('A*01:01')
    assert embedding.shape == (1024,), f"Wrong shape: {embedding.shape}"
    print(f"✓ ProtT5: {embedding.shape}")
    
    print("\n--- Testing Ankh Base encoding ---")
    encoder = AnkhEncoder(temp_file, model_variant='base', device='cpu')
    embedding = encoder.get_embedding('A*01:01')
    assert embedding.shape == (768,), f"Wrong shape: {embedding.shape}"
    print(f"✓ Ankh Base: {embedding.shape}")
    
    print("\n--- Testing Ankh Large encoding ---")
    encoder = AnkhEncoder(temp_file, model_variant='large', device='cpu')
    embedding = encoder.get_embedding('A*01:01')
    assert embedding.shape == (1536,), f"Wrong shape: {embedding.shape}"
    print(f"✓ Ankh Large: {embedding.shape}")
    
    print("\n" + "="*50)
    print("ALL ENCODERS WORKING CORRECTLY!")
    print("="*50)
    
finally:
    os.unlink(temp_file)
EOF
    
    python /tmp/test_encoder_function.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All encoders functional${NC}"
    else
        echo -e "${RED}✗ Encoder functionality test failed${NC}"
        exit 1
    fi
    
    rm /tmp/test_encoder_function.py
}

# Run pytest
run_tests() {
    echo ""
    echo "Step 5: Running test suite..."
    echo "=========================================="
    
    python -m pytest tests/test_prott5_encoder.py tests/test_ankh_encoder.py -v
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

# Main installation flow
main() {
    check_venv
    install_dependencies
    verify_packages
    test_model_downloads
    test_encoder_functionality
    run_tests
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}INSTALLATION COMPLETE!${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run the complete pipeline:"
    echo "   ./run_complete_pipeline_all_encoders.sh"
    echo ""
    echo "2. Or test individual encoders:"
    echo "   python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all"
    echo "   python -m hlaprotbert.scripts.generate_embeddings --encoder-type esm --all"
    echo "   python -m hlaprotbert.scripts.generate_embeddings --encoder-type prott5 --all"
    echo "   python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-base --all"
    echo "   python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-large --all"
    echo ""
    echo "3. Compare encoders:"
    echo "   python examples/multi_encoder_comparison.py --alleles A*01:01 A*02:01 --benchmark"
    echo ""
}

# Run main
main
