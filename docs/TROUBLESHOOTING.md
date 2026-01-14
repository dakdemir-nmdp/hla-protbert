# HLA-ProtBERT Troubleshooting Guide

This guide helps you resolve common issues when using HLA-ProtBERT.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Download Problems](#data-download-problems)
3. [Model Loading Errors](#model-loading-errors)
4. [Memory Issues](#memory-issues)
5. [GPU/CUDA Problems](#gpucuda-problems)
6. [Encoding Errors](#encoding-errors)
7. [Performance Issues](#performance-issues)
8. [Import Errors](#import-errors)

---

## Installation Issues

### Problem: `pip install -r requirements.txt` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=1.7.0
```

**Solutions:**
1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

2. Install PyTorch separately (for specific CUDA version):
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. Then install remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Problem: ModuleNotFoundError after installation

**Symptoms:**
```python
ModuleNotFoundError: No module named 'src'
```

**Solutions:**
1. Install package in development mode:
   ```bash
   pip install -e .
   ```

2. Or add project to PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/hla-protbert"
   ```

---

## Data Download Problems

### Problem: FTP connection timeout

**Symptoms:**
```
Error downloading from FTP: Connection timed out
```

**Solutions:**
1. Use GitHub fallback:
   ```python
   from hlaprotbert.data.imgt_downloader import IMGTDownloader
   
   downloader = IMGTDownloader(use_github_first=True)
   downloader.download_latest()
   ```

2. Increase timeout:
   ```python
   downloader = IMGTDownloader(ftp_timeout=120)  # 2 minutes
   ```

3. Check firewall/proxy settings - FTP port 21 must be open

### Problem: SSL Certificate Verification Failed

**Symptoms:**
```
SSLError: certificate verify failed
```

**Solutions:**
1. Disable SSL verification (use with caution):
   ```python
   encoder = ProtBERTEncoder(
       sequence_file="data/sequences.pkl",
       verify_ssl=False
   )
   ```

2. Update CA certificates:
   ```bash
   # macOS
   pip install --upgrade certifi
   
   # Linux
   sudo update-ca-certificates
   ```

---

## Model Loading Errors

### Problem: Transformers model download fails

**Symptoms:**
```
OSError: Can't load tokenizer for 'Rostlab/prot_bert'
```

**Solutions:**
1. Clear Hugging Face cache:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

2. Download models manually with authentication:
   ```python
   from transformers import AutoTokenizer, AutoModel
   from huggingface_hub import login
   
   login(token="your_hf_token")  # Get from https://huggingface.co/settings/tokens
   
   tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
   model = AutoModel.from_pretrained("Rostlab/prot_bert")
   ```

3. Use local model path:
   ```python
   encoder = ProtBERTEncoder(
       sequence_file="data/sequences.pkl",
       model_name="/path/to/local/prot_bert"
   )
   ```

### Problem: Model files corrupted

**Symptoms:**
```
RuntimeError: Error loading model weights
```

**Solutions:**
1. Re-download models:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--Rostlab--prot_bert
   ```

2. Check disk space:
   ```bash
   df -h ~/.cache/huggingface/
   ```

---

## Memory Issues

### Problem: Out of Memory (OOM) errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
# or
MemoryError: Unable to allocate array
```

**Solutions:**
1. Reduce batch size:
   ```python
   embeddings = encoder.batch_encode_alleles(
       alleles,
       batch_size=4  # Reduce from default 8
   )
   ```

2. Use CPU instead of GPU:
   ```python
   encoder = ProtBERTEncoder(
       sequence_file="data/sequences.pkl",
       device="cpu"
   )
   ```

3. Process in smaller chunks:
   ```python
   chunk_size = 100
   all_embeddings = {}
   
   for i in range(0, len(alleles), chunk_size):
       chunk = alleles[i:i+chunk_size]
       embeddings = encoder.batch_encode_alleles(chunk)
       all_embeddings.update(embeddings)
   ```

4. Clear cache periodically:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Problem: System runs out of RAM during processing

**Solutions:**
1. Monitor memory usage:
   ```python
   import psutil
   print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
   ```

2. Use streaming/generator patterns:
   ```python
   # Instead of loading all at once
   for allele in alleles:
       embedding = encoder.get_embedding(allele)
       # Process immediately
   ```

---

## GPU/CUDA Problems

### Problem: CUDA not detected

**Symptoms:**
```
torch.cuda.is_available() returns False
```

**Solutions:**
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Install correct PyTorch version:
   ```bash
   # Check CUDA version first
   nvidia-smi
   
   # Install matching PyTorch (example for CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Verify PyTorch CUDA:
   ```python
   import torch
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```

### Problem: CUDA version mismatch

**Solutions:**
1. Check versions:
   ```bash
   nvidia-smi  # System CUDA
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
   ```

2. Reinstall matching PyTorch

---

## Encoding Errors

### Problem: ValueError: No sequence found for allele

**Symptoms:**
```
ValueError: No sequence found for allele B*57:01:01:02
```

**Solutions:**
1. Use 2-field resolution:
   ```python
   embedding = encoder.get_embedding("B*57:01")  # Instead of B*57:01:01:02
   ```

2. Check available alleles:
   ```python
   print(list(encoder.sequences.keys())[:10])
   ```

3. Update IMGT database:
   ```python
   from hlaprotbert.data.imgt_downloader import IMGTDownloader
   
   downloader = IMGTDownloader()
   downloader.download_latest(force=True)
   ```

### Problem: Allele standardization issues

**Solutions:**
1. Use standardized format:
   ```python
   # Correct formats
   "A*01:01"
   "B*07:02"
   "DRB1*03:01"
   
   # Avoid
   "HLA-A*01:01"  # Will be auto-standardized
   "A0101"  # May not work without locus context
   ```

---

## Performance Issues

### Problem: Encoding is very slow

**Solutions:**
1. Enable GPU:
   ```python
   encoder = ProtBERTEncoder(
       sequence_file="data/sequences.pkl",
       device="cuda"  # or "auto"
   )
   ```

2. Increase batch size:
   ```python
   embeddings = encoder.batch_encode_alleles(
       alleles,
       batch_size=32  # Increase if you have enough memory
   )
   ```

3. Use cached embeddings:
   ```python
   # Embeddings are cached by default
   # First run: slow
   embeddings = encoder.batch_encode_alleles(alleles)
   
   # Subsequent runs: fast (uses cache)
   embeddings = encoder.batch_encode_alleles(alleles)
   ```

4. Use ESM instead of ProtBERT (faster inference):
   ```python
   from hlaprotbert.models.encoders.esm import ESMEncoder
   
   encoder = ESMEncoder(
       sequence_file="data/sequences.pkl",
       model_name="facebook/esm2_t33_650M_UR50D"
   )
   ```

---

## Import Errors

### Problem: Cannot import from src

**Symptoms:**
```python
ImportError: cannot import name 'ProtBERTEncoder' from 'src.models.encoders'
```

**Solutions:**
1. Verify installation:
   ```bash
   pip show hlaprotbert
   ```

2. Reinstall package:
   ```bash
   pip uninstall hlaprotbert
   pip install -e .
   ```

3. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

---

## Getting Additional Help

If your problem isn't covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/dakdemir-nmdp/hla-protbert/issues)
2. **Search documentation**: [Full Documentation](https://github.com/dakdemir-nmdp/hla-protbert/tree/main/docs)
3. **Open a new issue**: Provide:
   - Python version (`python --version`)
   - Package versions (`pip list | grep -E "torch|transformers|numpy"`)
   - Full error traceback
   - Minimal reproducible example
   - System information (OS, RAM, GPU)

## Quick Diagnostic Script

Run this to collect system information:

```python
import sys
import platform
import torch
import transformers
import numpy as np

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"CPU cores: {os.cpu_count()}")
```

Save this as `diagnostic.py` and share the output when reporting issues.
