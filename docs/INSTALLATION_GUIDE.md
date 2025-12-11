# HLA-ProtBERT Installation Guide

This guide provides step-by-step instructions for installing and verifying HLA-ProtBERT with all five protein language model encoders.

## Prerequisites

- Python 3.9 or higher
- 16GB+ RAM recommended (for large models)
- GPU optional but recommended for faster encoding
- Internet connection for downloading models

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hla-protbert.git
cd hla-protbert
```

### 2. Create Virtual Environment

**IMPORTANT**: Always use a virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

You should see `(venv)` at the beginning of your terminal prompt.

### 3. Run Setup and Verification

We provide an automated script that:
- Installs all dependencies
- Verifies package versions
- Downloads all 5 encoder models
- Tests each encoder with a sample sequence
- Runs the full test suite

```bash
./setup_and_verify.sh
```

This will take 5-15 minutes depending on your internet speed (models are ~2-5GB).

**Expected output:**
```
==========================================
HLA-ProtBERT Setup and Verification
==========================================

Step 1: Checking virtual environment
✓ Virtual environment detected

Step 2: Installing dependencies
[... pip output ...]
✓ Dependencies installed

Step 3: Verifying package versions
✓ Package versions correct

Step 4: Testing model downloads
Downloading ProtBERT model...
✓ ProtBERT model downloaded
[... similar for other models ...]

Step 5: Testing encoder functionality
Testing ProtBERT encoder...
✓ ProtBERT encoder works
[... similar for other encoders ...]

Step 6: Running test suite
[... pytest output ...]
✓ All tests passed (30/30)

==========================================
Setup Complete!
==========================================
```

### 4. Verify Installation Manually (Optional)

If you want to verify installation step-by-step:

```bash
# Check Python version
python --version  # Should be 3.9+

# Install dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentencepiece; print(f'SentencePiece: {sentencepiece.__version__}')"

# Test each encoder
python -c "
from src.models.encoders import ProtBERTEncoder
encoder = ProtBERTEncoder()
emb = encoder.get_embedding('A*01:01')
print(f'ProtBERT works! Embedding shape: {emb.shape}')
"

python -c "
from src.models.encoders import ESMEncoder
encoder = ESMEncoder()
emb = encoder.get_embedding('A*01:01')
print(f'ESM-2 works! Embedding shape: {emb.shape}')
"

python -c "
from src.models.encoders import ProtT5Encoder
encoder = ProtT5Encoder()
emb = encoder.get_embedding('A*01:01')
print(f'ProtT5 works! Embedding shape: {emb.shape}')
"

python -c "
from src.models.encoders import AnkhEncoder
encoder = AnkhEncoder(model_variant='base')
emb = encoder.get_embedding('A*01:01')
print(f'Ankh Base works! Embedding shape: {emb.shape}')
"

python -c "
from src.models.encoders import AnkhEncoder
encoder = AnkhEncoder(model_variant='large')
emb = encoder.get_embedding('A*01:01')
print(f'Ankh Large works! Embedding shape: {emb.shape}')
"
```

### 5. Run the Complete Pipeline

Once verification passes, run the full pipeline to generate embeddings for all encoders:

```bash
./run_complete_pipeline_all_encoders.sh
```

This will:
1. Download HLA sequence data from IMGT/HLA
2. Generate embeddings with all 5 encoders
3. Run locus-specific analysis
4. Generate visualizations (t-SNE, UMAP, PCA)
5. Create summary statistics

**Pipeline duration**: 30-90 minutes depending on hardware (GPU recommended).

## Troubleshooting

### "Not in a virtual environment" Warning

```bash
# Make sure you activated the venv
source venv/bin/activate

# You should see (venv) in your prompt
```

### "Model download failed" Error

- Check internet connection
- Verify you have enough disk space (~10GB for all models)
- Try running again (downloads resume automatically)
- If behind a proxy, set `HF_HUB_OFFLINE=0`

### "SSL: CERTIFICATE_VERIFY_FAILED" when contacting Hugging Face

Some corporate networks use SSL interception which breaks the default certificate
validation during Hugging Face downloads. You have two options:

1. Install your organization's root certificate so Python trusts the proxy (recommended).
2. As a temporary workaround, disable SSL verification for the downloader:
   ```bash
   # Environment variable works for every helper script
   export HLA_DISABLE_SSL_VERIFY=1

   # Example: run the full pipeline with SSL verification disabled
   ./run_complete_pipeline_all_encoders.sh --disable-ssl-verify

   # Or when invoking the generator directly
   python scripts/generate_embeddings.py --encoder-type ankh-base --all --disable-ssl-verify
   ```

   You can also set `network.verify_ssl: false` in your config file if you always
   need this behavior.

> ⚠️ Disabling SSL verification should only be done on trusted networks because
> it removes certificate validation for Hugging Face model downloads.

### "I can't reach Hugging Face at all" (use the pip `ankh` package)

If your network policies block Hugging Face entirely, install the official
[Ankh package](https://github.com/agemagician/Ankh) and tell the tools to use it:

```bash
python -m pip install ankh

# CLI flag (applied per run)
python scripts/generate_embeddings.py --encoder-type ankh-base --all --ankh-backend ankh

# Pipeline helper
./run_complete_pipeline_all_encoders.sh --disable-ssl-verify  # optional

# config.yml
model:
  ankh_backend: ankh
```

When `ankh_backend` is set to `auto` (default), the encoder now tries Hugging
Face first and automatically falls back to the local `ankh` loaders whenever the
download fails. This works well in combination with the SSL-disable flag above.

### "CUDA out of memory" Error

For GPU memory issues:
- Use CPU instead: `export CUDA_VISIBLE_DEVICES=""`
- Reduce batch size in encoder initialization
- Process loci one at a time

### "Transformers version too old" Error

```bash
pip install --upgrade transformers>=4.30.0
```

### "sentencepiece not found" Error

```bash
pip install sentencepiece>=0.1.99
```

### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/hla-protbert

# Install in development mode
pip install -e .
```

## What Gets Downloaded?

The setup downloads these models from Hugging Face:

| Model | Size | Provider | Params |
|-------|------|----------|--------|
| ProtBERT | ~2GB | Rostlab | 420M |
| ESM-2 | ~2.5GB | Facebook AI | 650M |
| ProtT5 | ~5GB | Rostlab | 1.3B |
| Ankh Base | ~200MB | ElnaggarLab | 50M |
| Ankh Large | ~2.5GB | ElnaggarLab | 650M |

**Total**: ~12GB download + ~15GB on disk (with cache)

## Directory Structure After Setup

```
hla-protbert/
├── venv/                    # Virtual environment
├── data/
│   ├── raw/                 # Downloaded IMGT data
│   ├── processed/           # Parsed sequences
│   ├── embeddings/          # Generated embeddings
│   │   ├── protbert/
│   │   ├── esm/
│   │   ├── prott5/
│   │   └── ankh/
│   │       ├── base/
│   │       └── large/
│   └── analysis/            # Analysis results
├── .cache/                  # Model cache (Hugging Face)
└── htmlcov/                 # Test coverage reports
```

## Quick Test After Installation

```bash
# Test basic encoding
python examples/basic_encoding.py

# Compare all encoders
python examples/multi_encoder_comparison.py \
    --alleles A*01:01 A*02:01 B*07:02 \
    --benchmark

# Expected output:
# Encoder        | Embed Dim | Time (s)
# ---------------|-----------|----------
# ProtBERT       | 768       | 0.124
# ESM-2          | 1280      | 0.156
# ProtT5         | 1024      | 0.189
# Ankh Base      | 768       | 0.098
# Ankh Large     | 1536      | 0.211
```

## Next Steps

1. **Explore Examples**: Check out `examples/` directory
2. **Read API Docs**: See `docs/API_REFERENCE.md`
3. **Run Notebooks**: Try `notebooks/hla_protbert_demo.ipynb`
4. **Customize Pipeline**: Modify `scripts/generate_embeddings.py`

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory  
- **Issues**: Report bugs on GitHub Issues
- **Questions**: See `docs/TROUBLESHOOTING.md`

## Verification Checklist

After installation, verify:

- [ ] Virtual environment activated (`(venv)` in prompt)
- [ ] All dependencies installed (`pip list | grep transformers`)
- [ ] All 5 encoders import successfully
- [ ] Each encoder can generate embeddings
- [ ] Test suite passes (30/30 tests)
- [ ] Pipeline script runs without errors
- [ ] Embeddings saved in `data/embeddings/`

---

**Installation Time**: ~15-20 minutes (with good internet)
**Disk Space Required**: ~15GB
**RAM Required**: 8GB minimum, 16GB recommended
**GPU**: Optional but recommended for large-scale encoding
