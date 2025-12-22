# HLA-ProtBERT Benchmarks

This directory contains benchmarking tools and results for comparing the performance of different HLA protein encoders.

## Quick Start

```bash
# Quick test (3 runs, 10 alleles, batch size 8)
python benchmarks/benchmark_encoders.py --quick

# Full benchmark (all encoders, multiple batch sizes)
python benchmarks/benchmark_encoders.py

# Benchmark specific encoder
python benchmarks/benchmark_encoders.py --encoder protbert --n-runs 20
```

## What's Benchmarked

### Encoders
- **ProtBERT**: BERT-based, 420M params, 768-dim embeddings
- **ESM-2**: RoBERTa-based, 650M params, 1280-dim embeddings  
- **ProtT5**: T5-based, 1.3B params, 1024-dim embeddings
- **Ankh Base**: Custom, 50M params, 768-dim embeddings
- **Ankh Large**: Custom, 650M params, 1536-dim embeddings

### Metrics
1. **Single Allele Encoding Time**
   - Mean, std, min, max, median (seconds)
   - Measured over multiple runs

2. **Batch Encoding Throughput**
   - Alleles processed per second
   - Tested with different batch sizes (8, 16, 32)

3. **System Information**
   - Platform, CPU, GPU details
   - Python and PyTorch versions
   - Reproducibility metadata

## Results

### Example Output

```
BENCHMARK SUMMARY
================================================================================

ProtBERT:
  Embedding Dimension: 768
  Single Allele: 0.4523s ± 0.0234s
  batch_8: 18.45 alleles/sec
  batch_16: 22.13 alleles/sec
  batch_32: 24.67 alleles/sec

ESM-2:
  Embedding Dimension: 1280
  Single Allele: 0.6012s ± 0.0312s
  batch_8: 14.23 alleles/sec
  batch_16: 16.89 alleles/sec

...
```

Results are saved to `benchmarks/results.json` with full details including:
- System specifications
- Test configuration  
- Per-encoder statistics
- Embedding dimensions

## Running Benchmarks

### Prerequisites

```bash
# Ensure HLA sequences are downloaded
python scripts/update_imgt.py

# Install all required models will happen automatically on first run
# (Models will be downloaded to HuggingFace cache, ~15GB total)
```

### Basic Usage

```bash
# Default: Benchmark all encoders with 10 runs
python benchmarks/benchmark_encoders.py

# Output to specific file
python benchmarks/benchmark_encoders.py --output my_results.json

# Use specific locus
python benchmarks/benchmark_encoders.py --locus B --n-alleles 30
```

### Advanced Options

```bash
# More runs for higher precision
python benchmarks/benchmark_encoders.py --n-runs 20

# Different batch sizes
python benchmarks/benchmark_encoders.py --batch-sizes 4 8 16 32 64

# Force CPU (useful for comparing CPU vs GPU)
python benchmarks/benchmark_encoders.py --device cpu

# Benchmark single encoder
python benchmarks/benchmark_encoders.py --encoder ankh-base
```

## Interpreting Results

### Single Allele Time
- Lower is better
- Includes model inference and embedding extraction
- **Does NOT include** model loading time (one-time cost)

### Throughput (alleles/sec)
- Higher is better
- Useful for large-scale encoding tasks
- Scales with batch size (up to memory limit)

### Typical Performance (Example Hardware)

**Apple M1 MacBook Pro (CPU)**
| Encoder | Single Allele | Batch (8) Throughput |
|---------|--------------|---------------------|
| Ankh Base | ~0.15s | ~50 alleles/sec |
| ProtBERT | ~0.45s | ~18 alleles/sec |
| ESM-2 | ~0.60s | ~14 alleles/sec |

**NVIDIA A100 GPU**
| Encoder | Single Allele | Batch (32) Throughput |
|---------|--------------|---------------------|
| Ankh Base | ~0.02s | ~500 alleles/sec |
| ProtBERT | ~0.05s | ~200 alleles/sec |
| ESM-2 | ~0.08s | ~150 alleles/sec |

*Note: Actual performance depends on hardware, sequence length, and system load*

## Reproducibility

### System Information Captured
- Operating system and version
- CPU model and thread count
- GPU availability and model
- Python version
- PyTorch version
- CUDA version (if applicable)

### Methodology
1. **Warmup**: One encoding run to load model weights into memory
2. **Measurement**: Multiple timed runs (default: 10)
3. **Statistics**: Mean, std, min, max, median computed
4. **Batch Tests**: Multiple batch sizes tested independently

### Variance Sources
- System background processes
- CPU/GPU thermal throttling
- Network latency (first model download only)
- Random initialization (minimal impact)

**Best Practices**:
- Close unnecessary applications
- Run multiple times and average
- Use same hardware for comparisons
- Report system specifications

## Output Format

Results are saved as JSON with this structure:

```json
{
  "metadata": {
    "timestamp": "2025-12-22 10:00:00",
    "system_info": {
      "platform": "macOS-14.0-arm64",
      "processor": "arm",
      "python_version": "3.10.0",
      "cuda_available": false,
      ...
    },
    "test_config": {
      "locus": "A",
      "n_alleles": 50,
      "n_runs": 10,
      "batch_sizes": [8, 16, 32]
    }
  },
  "results": [
    {
      "encoder_name": "ProtBERT",
      "embedding_dim": 768,
      "single_allele": {
        "mean_sec": 0.4523,
        "std_sec": 0.0234,
        ...
      },
      "batch_encoding": {
        "batch_8": {
          "mean_sec": 4.32,
          "throughput_alleles_per_sec": 18.45
        },
        ...
      },
      "status": "success"
    },
    ...
  ]
}
```

## Adding Custom Benchmarks

To add custom benchmarks, extend `benchmark_encoders.py`:

```python
def benchmark_custom_metric(encoder, alleles, **kwargs):
    """Your custom benchmark function."""
    # Implement your metric
    return results

# Add to main():
custom_results = benchmark_custom_metric(encoder, test_alleles)
all_results['custom_metric'] = custom_results
```

## Troubleshooting

### "RuntimeError: CUDA out of memory"
```bash
# Reduce batch size
python benchmarks/benchmark_encoders.py --batch-sizes 4 8

# Use CPU
python benchmarks/benchmark_encoders.py --device cpu
```

### "Sequence file not found"
```bash
# Download HLA sequences first
python scripts/update_imgt.py
```

### Model download issues
```bash
# Check internet connection
# Models download automatically on first run
# Cache location: ~/.cache/huggingface/
```

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{hla_protbert_benchmarks,
  author = {Deniz Akdemir},
  title = {HLA-ProtBERT Benchmarks},
  year = {2025},
  url = {https://github.com/dakdemir-nmdp/hla-protbert}
}
```

## Future Improvements

Planned additions:
- [ ] Memory usage profiling
- [ ] Fine-tuned model benchmarks
- [ ] Comparison with raw transformers
- [ ] Domain-specific evaluation metrics
- [ ] Visualization dashboard

## Questions?

- **Issues**: https://github.com/dakdemir-nmdp/hla-protbert/issues
- **Email**: dakdemir@nmdp.org
