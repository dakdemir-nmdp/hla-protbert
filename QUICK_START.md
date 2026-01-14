# Quick Start: Install and Run HLA-ProtBERT

## For Users in a Hurry

```bash
# 1. Activate your virtual environment (REQUIRED)
source venv/bin/activate

# 2. Install and verify everything
./setup_and_verify.sh

# 3. Run complete pipeline for all encoders
./run_complete_pipeline_all_encoders.sh

# 4. Compare encoder outputs
python examples/multi_encoder_comparison.py \
    --alleles A*01:01 A*02:01 B*07:02 \
    --benchmark
```

**Total time**: ~30-60 minutes (mostly model downloads)

---

## Step-by-Step Commands

### Before You Start
```bash
# Navigate to the project directory
cd path/to/hla-protbert

# Activate venv (you should see (venv) in your prompt after this)
source venv/bin/activate
```

### Step 1: Setup and Verify (15-20 min)
```bash
./setup_and_verify.sh
```

**What this does:**
- ✓ Checks you're in a venv
- ✓ Installs all Python dependencies from `requirements.txt`
- ✓ Downloads all 5 encoder models (~12GB)
- ✓ Tests each encoder with a sample sequence
- ✓ Runs full test suite (30 tests)

**Expected final output:**
```
✓ All tests passed (30/30)

==========================================
Setup Complete!
==========================================
All encoders are installed and working correctly.
```

### Step 2: Run Complete Pipeline (30-60 min)
```bash
./run_complete_pipeline_all_encoders.sh
```

**What this does:**
- Downloads HLA sequences from IMGT/HLA database
- Generates embeddings with all 5 encoders:
  - ProtBERT (1024-dim)
  - ESM-2 (1280-dim)
  - ProtT5 (1024-dim)
  - Ankh Base (768-dim)
  - Ankh Large (1536-dim)
- Runs locus-specific analysis (A, B, C, DRB1, etc.)
- Creates visualizations (t-SNE, UMAP, PCA plots)
- Generates summary statistics

**Expected final output:**
```
==========================================
Pipeline Complete!
==========================================
✓ All encoders have been run successfully!

HLA-ProtBERT Pipeline Summary
==============================================================
Embedding Statistics:
--------------------------------------------------------------
ProtBERT    : 15234 alleles,  768-dim embeddings
ESM-2       : 15234 alleles, 1280-dim embeddings
ProtT5      : 15234 alleles, 1024-dim embeddings
Ankh Base   : 15234 alleles,  768-dim embeddings
Ankh Large  : 15234 alleles, 1536-dim embeddings
```

### Step 3: Verify Results
```bash
# Check embeddings were created
ls -lh data/embeddings/*/hla_embeddings.pkl

# Expected output:
# data/embeddings/protbert/hla_embeddings.pkl  (~45MB)
# data/embeddings/esm/hla_embeddings.pkl       (~75MB)
# data/embeddings/prott5/hla_embeddings.pkl    (~60MB)
# data/embeddings/ankh/base/hla_embeddings.pkl (~45MB)
# data/embeddings/ankh/large/hla_embeddings.pkl (~90MB)
```

---

## What If Something Goes Wrong?

### "Not in a virtual environment"
```bash
source venv/bin/activate
# You MUST see (venv) in your prompt
```

### "Model download failed"
```bash
# Check internet connection, then retry
./setup_and_verify.sh
# Downloads resume automatically
```

### "Tests failed"
```bash
# Run tests with verbose output
pytest tests/ -v --tb=short

# If specific encoder fails, test individually
python -c "
from hlaprotbert.models.encoders import ProtBERTEncoder
encoder = ProtBERTEncoder()
print('ProtBERT works:', encoder.get_embedding('A*01:01').shape)
"
```

### "Pipeline crashed midway"
```bash
# Pipeline is resumable - just run again
# It will skip already-generated embeddings
./run_complete_pipeline_all_encoders.sh

# When prompted "Regenerate embeddings? (y/n)", press 'n' to skip
```

---

## After Installation: Usage Examples

### Example 1: Encode a Single Allele
```python
from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder

# Pick any encoder
encoder = ProtT5Encoder()
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")  # (1024,)
```

### Example 2: Encode Multiple Alleles
```python
from hlaprotbert.models.encoders import AnkhEncoder

encoder = AnkhEncoder(model_variant='large')
alleles = ["A*01:01", "A*02:01", "B*07:02", "B*08:01"]
embeddings = encoder.batch_encode_alleles(alleles)

for allele, embedding in embeddings.items():
    print(f"{allele}: {embedding.shape}")
```

### Example 3: Compare All Encoders
```bash
python examples/multi_encoder_comparison.py \
    --alleles A*01:01 A*02:01 A*03:01 \
    --benchmark \
    --visualize
```

### Example 4: Load Pre-Generated Embeddings
```python
import pickle

# Load embeddings from pipeline run
with open('data/embeddings/prott5/hla_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print(f"Loaded {len(embeddings)} alleles")
print(f"A*01:01 embedding: {embeddings['A*01:01'].shape}")
```

---

## File Locations After Pipeline

```
data/
├── processed/
│   └── hla_sequences.pkl           # Parsed HLA sequences
├── embeddings/
│   ├── protbert/
│   │   └── hla_embeddings.pkl      # ProtBERT embeddings
│   ├── esm/
│   │   └── hla_embeddings.pkl      # ESM-2 embeddings
│   ├── prott5/
│   │   └── hla_embeddings.pkl      # ProtT5 embeddings
│   └── ankh/
│       ├── base/
│       │   └── hla_embeddings.pkl  # Ankh Base embeddings
│       └── large/
│           └── hla_embeddings.pkl  # Ankh Large embeddings
└── analysis/
    └── locus_embeddings/
        ├── class1/                  # Class I (A, B, C)
        └── class2/                  # Class II (DRB1, DQB1, DPB1)
```

---

## Common Questions

**Q: Do I need a GPU?**  
A: No, but it's 5-10x faster. CPU works fine for small datasets.

**Q: How long does the full pipeline take?**  
A: 30-60 minutes on modern hardware (depends on CPU/GPU and internet speed).

**Q: Can I run just one encoder?**  
A: Yes! Use `scripts/generate_embeddings.py --encoder-type protbert`

**Q: What if I only want Class I alleles?**  
A: Modify the pipeline script or use `--loci A B C` flag in scripts.

**Q: Can I use my own sequences?**  
A: Yes! See `docs/API_REFERENCE.md` for custom sequence encoding.

---

## Verification Checklist

Before running the pipeline, verify:

- [x] `(venv)` appears in your terminal prompt
- [x] `./setup_and_verify.sh` completed successfully
- [x] All 30 tests passed
- [x] All 5 encoders imported and tested

After running the pipeline, verify:

- [x] 5 embedding files exist in `data/embeddings/`
- [x] Each embedding file is 40-90MB
- [x] Summary shows ~15k alleles encoded
- [x] No error messages in terminal output

---

## Need Help?

- **Full Documentation**: `docs/INSTALLATION_GUIDE.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Examples**: `examples/` directory
- **Issues**: Report on GitHub

**Installation complete? Start here:** `docs/QUICK_START_NEW_ENCODERS.md`
