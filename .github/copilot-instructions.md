# HLA-ProtBERT Copilot Instructions

## Project Overview
This is a bioinformatics framework for encoding HLA (Human Leukocyte Antigen) protein sequences into high-dimensional embeddings using transformer models (ProtBERT, ESM). The system supports transplant compatibility analysis, allele similarity matching, and immunogenetics research.

## Architecture Patterns

### Core Component Structure
- **`src/models/encoder.py`**: Base `HLAEncoder` class with caching, fallback mechanisms, and sequence standardization
- **`src/models/encoders/`**: Concrete implementations (`ProtBERTEncoder`, `ESMEncoder`) that inherit from base
- **`src/data/`**: IMGT/HLA database management with FTP fallback to GitHub for cloud environments  
- **`src/analysis/`**: Downstream analysis tools (matching, visualization, metrics)

### Data Flow Architecture
1. **Download**: `scripts/update_imgt.py` → IMGT/HLA database → `data/raw/`
2. **Process**: Parse FASTA → standardized sequences → `data/processed/hla_sequences.pkl`
3. **Encode**: Batch encoding → cached embeddings → `data/embeddings/{protbert,esm}/`
4. **Analyze**: Locus-specific analysis → `data/analysis/locus_embeddings/{class1,class2}/`

### Key Design Patterns
- **Encoder Pattern**: All encoders implement `_encode_sequence()` and inherit caching/fallback logic
- **Graceful Degradation**: Missing dependencies (py-ard, transformers) log warnings but don't crash
- **Locus-Specific Processing**: Class I (A,B,C) and Class II (DRB1,DQB1,DPB1) handled separately
- **Embedding Caching**: Pickle-based caching with encoder-specific directories to avoid recomputation

## Critical Developer Workflows

### Initial Setup & Pipeline
```bash
# Always run complete pipeline for new environments
./run_complete_pipeline.sh

# Or manual step-by-step:
python scripts/update_imgt.py --verbose
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose
```

### Testing Strategy
- **Mock Encoders**: Use `MockHLAEncoder` pattern in tests (see `tests/test_encoder.py`)
- **Temporary Files**: Tests use `tempfile` for sequence files and cache directories
- **Batch Testing**: Run `pytest` from project root, tests expect `src/` structure

### Development Dependencies
- **Transformers**: Required for ProtBERT/ESM models - install with `pip install transformers>=4.0.0`
- **Optional Libraries**: `py-ard` (HLA nomenclature), `umap-learn` (visualization), `reportlab` (PDF reports)
- **GPU Support**: Use `torch>=1.7.0` with CUDA for faster encoding

## Project-Specific Conventions

### HLA Nomenclature
- Standard format: `A*01:01`, `B*07:02`, `DRB1*01:01` 
- Use `py-ard` library for allele resolution when available
- Fallback to string matching for allele normalization

### File Organization
- **Embeddings**: Organized by encoder type in `data/embeddings/{encoder}/`
- **Analysis**: Locus-specific results in `data/analysis/locus_embeddings/{class1,class2}/`
- **Caching**: Pickle files with `.pkl` extension, always check existence before loading

### Sequence Processing
- **Protein Sequences**: Space-separated amino acids for transformer input: `"M A V M A P R T L"`
- **Max Length**: 512 tokens for BERT models, handle truncation gracefully
- **Batch Processing**: Use `batch_encode_alleles()` for multiple sequences to leverage GPU efficiently

### Configuration Management
- **Config Pattern**: Use `ConfigManager` class for settings, supports JSON/YAML
- **Environment Variables**: Respect SSL verification settings (`verify_ssl=False` for internal networks)
- **Device Selection**: Auto-detect GPU availability, fallback to CPU

## Integration Points

### External Dependencies
- **IMGT/HLA Database**: Primary source via FTP, GitHub fallback (`ANHIG/IMGTHLA` repo)
- **Hugging Face Models**: `Rostlab/prot_bert`, `facebook/esm2_t33_650M_UR50D`
- **Jupyter Notebooks**: Integration with analysis notebooks in `notebooks/`

### Model Loading Patterns
```python
# Standard encoder initialization pattern
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings/protbert"
)
```

### Error Handling
- **Missing Models**: Log warnings for missing transformers, continue with degraded functionality
- **Network Issues**: IMGT downloader has FTP timeout handling and GitHub fallback
- **Sequence Errors**: Invalid alleles are logged and skipped, don't halt batch processing

## Key Files to Reference
- **`src/models/encoder.py`**: Core encoder interface and caching logic
- **`scripts/generate_embeddings.py`**: Primary workflow for embedding generation  
- **`run_complete_pipeline.sh`**: Complete setup and analysis pipeline
- **`src/analysis/matching.py`**: Donor-recipient compatibility analysis patterns
- **`tests/test_*`**: Testing patterns for encoders and utilities