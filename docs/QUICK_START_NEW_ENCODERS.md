# Quick Start: Using the New ProtT5 and Ankh Encoders

## Installation

Ensure you have the updated dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `transformers>=4.30.0` (updated for ProtT5 and Ankh support)
- `sentencepiece>=0.1.99` (required for T5 tokenizer)

## 1. Basic Usage

### ProtT5 Encoder

```python
from hlaprotbert.models.encoders import ProtT5Encoder

# Initialize encoder
encoder = ProtT5Encoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    device="auto"  # Automatically detects GPU/CPU
)

# Encode a single allele
embedding = encoder.get_embedding("A*01:01")
print(f"ProtT5 embedding shape: {embedding.shape}")  # (1024,)

# Batch encode multiple alleles
alleles = ["A*01:01", "A*02:01", "A*03:01", "B*07:02"]
embeddings = encoder.batch_encode_alleles(alleles, batch_size=4)
print(f"Encoded {len(embeddings)} alleles")
```

### Ankh Encoder (Base - Fast)

```python
from hlaprotbert.models.encoders import AnkhEncoder

# Initialize with base model (50M params, very fast)
encoder = AnkhEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_variant="base"
)

# Encode alleles
embedding = encoder.get_embedding("A*01:01")
print(f"Ankh Base embedding shape: {embedding.shape}")  # (768,)
```

### Ankh Encoder (Large - Accurate)

```python
from hlaprotbert.models.encoders import AnkhEncoder

# Initialize with large model (650M params, higher accuracy)
encoder = AnkhEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_variant="large"
)

# Encode alleles
embedding = encoder.get_embedding("A*01:01")
print(f"Ankh Large embedding shape: {embedding.shape}")  # (1536,)
```

## 2. Command-Line Usage

### Generate embeddings with ProtT5

```bash
# Download sequences first (if not already done)
python -m hlaprotbert.scripts.update_imgt --verbose

# Generate ProtT5 embeddings for all alleles
python -m hlaprotbert.scripts.generate_embeddings --encoder-type prott5 --all --verbose

# Generate for specific locus only
python -m hlaprotbert.scripts.generate_embeddings --encoder-type prott5 --locus A --all
```

### Generate embeddings with Ankh

```bash
# Ankh Base (fast)
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-base --all --verbose

# Ankh Large (accurate)
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-large --locus A --all
```

### Handling SSL certificate issues on locked-down networks

If Hugging Face downloads fail with `SSL: CERTIFICATE_VERIFY_FAILED`, you can
temporarily disable certificate verification:

```bash
# Environment variable applies to every invocation in the session
export HLA_DISABLE_SSL_VERIFY=1

# Pass the flag explicitly when running the generator
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-base --all --disable-ssl-verify
```

You can also set `network.verify_ssl: false` inside your config file if the same
setting should apply to every run. Remember to re-enable SSL verification when
you are on a trusted network.

When you don't want to rely on Hugging Face at all, install the native Ankh
package and let the encoder use it directly:

```bash
python -m pip install ankh
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-base --all --ankh-backend ankh
```

Setting `model.ankh_backend: ankh` in your config makes this the default for
every run. With `ankh_backend: auto` (default) the loader will try Hugging Face
first and automatically fall back to the local package.

## 3. Model Comparison

Compare all encoders side-by-side:

```bash
python examples/multi_encoder_comparison.py \
    --alleles A*01:01 A*02:01 A*03:01 \
    --compare-alleles \
    --ensemble \
    --benchmark \
    --verbose
```

This will:
- Encode alleles with all available encoders
- Compare inter-allele similarities
- Create ensemble embeddings
- Benchmark performance

## 4. Choosing the Right Encoder

### Quick Decision Tree

**Need fast inference for production?**
→ Use **Ankh Base** (50M params, 768-dim, very fast)

**Need proven baseline for research?**
→ Use **ProtBERT** (420M params, 768-dim, well-established)

**Need highest accuracy?**
→ Use **ESM-2** (650M params, 1280-dim, state-of-the-art)

**Building ensemble models?**
→ Add **ProtT5** (1.3B params, 1024-dim, complementary T5 architecture)

**Need balance of speed and accuracy?**
→ Use **Ankh Large** (650M params, 1536-dim, modern design)

## 5. Advanced: Ensemble Embeddings

```python
from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder
import numpy as np
from sklearn.preprocessing import normalize

# Initialize all encoders
protbert = ProtBERTEncoder("./data/processed/hla_sequences.pkl")
esm = ESMEncoder("./data/processed/hla_sequences.pkl")
prott5 = ProtT5Encoder("./data/processed/hla_sequences.pkl")
ankh = AnkhEncoder("./data/processed/hla_sequences.pkl", model_variant="large")

# Get embeddings
allele = "A*01:01"
emb_protbert = normalize(protbert.get_embedding(allele).reshape(1, -1))[0]  # 768-dim
emb_esm = normalize(esm.get_embedding(allele).reshape(1, -1))[0]            # 1280-dim
emb_prott5 = normalize(prott5.get_embedding(allele).reshape(1, -1))[0]      # 1024-dim
emb_ankh = normalize(ankh.get_embedding(allele).reshape(1, -1))[0]          # 1536-dim

# Concatenate normalized embeddings
ensemble_embedding = np.concatenate([emb_protbert, emb_esm, emb_prott5, emb_ankh])
print(f"Ensemble shape: {ensemble_embedding.shape}")  # (4608,)
```

## 6. Performance Tips

### GPU Acceleration

All encoders automatically use GPU if available:

```python
encoder = ProtT5Encoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    device="cuda"  # Explicit GPU
)
```

### Batch Processing

Use batch encoding for better performance:

```python
# Read alleles from file
with open("alleles.txt") as f:
    alleles = [line.strip() for line in f]

# Batch encode with optimal batch size
encoder = AnkhEncoder(
    "./data/processed/hla_sequences.pkl",
    model_variant="base"
)

# Ankh Base can handle larger batches
embeddings = encoder.batch_encode_alleles(alleles, batch_size=32)

# ProtT5 needs smaller batches (larger model)
encoder_t5 = ProtT5Encoder("./data/processed/hla_sequences.pkl")
embeddings_t5 = encoder_t5.batch_encode_alleles(alleles, batch_size=4)
```

### Caching

Embeddings are automatically cached. To regenerate:

```python
# Force regeneration (ignores cache)
embedding = encoder.get_embedding("A*01:01", force=True)

# Or in batch mode
embeddings = encoder.batch_encode_alleles(alleles, force=True)
```

## 7. Troubleshooting

### Model Download Issues

If models fail to download:

```python
# Set HuggingFace token (if using gated models)
encoder = ProtT5Encoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    hf_token="your_token_here"
)
```

### Out of Memory

Reduce batch size:

```python
# Default batch sizes:
# - Ankh Base: 16 (can increase to 32 on GPU)
# - ProtBERT: 8
# - ESM-2: 8
# - Ankh Large: 8
# - ProtT5: 4 (largest model)

# If OOM, reduce batch size
embeddings = encoder.batch_encode_alleles(alleles, batch_size=2)
```

Or use CPU:

```python
encoder = ProtT5Encoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    device="cpu"
)
```

## 8. Next Steps

- **Read the README**: See full model comparison and recommendations
- **Check ADR-001**: Learn about planned structure-aware encoders
- **Run examples**: Try `examples/multi_encoder_comparison.py`
- **Read documentation**: See `docs/API_REFERENCE.md` for full API

## Common Workflows

### Workflow 1: Quick Encoding

```bash
# 1. Download data
python -m hlaprotbert.scripts.update_imgt

# 2. Encode with fastest model
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-base --all

# 3. Use in Python
python examples/basic_encoding.py --alleles A*01:01 A*02:01 --encoder-type ankh-base
```

### Workflow 2: High-Accuracy Research

```bash
# 1. Download data
python -m hlaprotbert.scripts.update_imgt

# 2. Encode with all models
python -m hlaprotbert.scripts.generate_embeddings --encoder-type protbert --all
python -m hlaprotbert.scripts.generate_embeddings --encoder-type esm --all
python -m hlaprotbert.scripts.generate_embeddings --encoder-type prott5 --all
python -m hlaprotbert.scripts.generate_embeddings --encoder-type ankh-large --all

# 3. Compare and create ensembles
python examples/multi_encoder_comparison.py --alleles A*01:01 A*02:01 --ensemble
```

### Workflow 3: Production Deployment

```python
from hlaprotbert.models.encoders import AnkhEncoder

# Use Ankh Base for production (fastest, efficient)
encoder = AnkhEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    model_variant="base",
    cache_dir="./production_cache",
    device="cuda"  # Use GPU if available
)

# Pre-generate embeddings for all alleles
all_alleles = list(encoder.sequences.keys())
embeddings = encoder.batch_encode_alleles(all_alleles, batch_size=32)

print(f"Generated {len(embeddings)} embeddings for production use")
```

## Resources

- **Full Documentation**: `docs/API_REFERENCE.md`
- **Model Details**: See README.md "Choosing the Right Encoder" section
- **Future Plans**: `docs/architecture/ADR-001-hybrid-embeddings.md`
- **Examples**: `examples/` directory
- **Tests**: `tests/test_prott5_encoder.py`, `tests/test_ankh_encoder.py`

## Support

For issues or questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Review test files for usage examples
3. See ADR-001 for planned features
4. Open an issue on GitHub

---

**Happy encoding!** 🧬🚀
