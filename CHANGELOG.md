# Changelog

All notable changes to HLA-ProtBERT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-14

### 🎉 First Public Release

This is the first public release of HLA-ProtBERT, a production-ready framework for encoding HLA alleles using state-of-the-art protein language models.

### Features

- **Multiple Protein Language Model Encoders**
  - ProtBERT (BERT-based, 420M params, 1024-dim embeddings)
  - ESM-2 (RoBERTa-based, 650M params, 1280-dim embeddings)
  - ProtT5 (T5-based, 1.3B params, 1024-dim embeddings)
  - Ankh Base (50M params, 768-dim embeddings) - optimized for fast inference
  - Ankh Large (650M params, 1536-dim embeddings) - balanced speed/accuracy

- **IMGT/HLA Database Integration**
  - Automated downloading and updating of HLA sequence data
  - Version tracking for reproducibility
  - Support for all major HLA loci (A, B, C, DRB1, DQB1, DPB1)

- **HLA-Specific Features**
  - Allele name standardization and resolution
  - Peptide binding region (PBR) extraction
  - Locus-specific encoding
  - ARD (Antigen Recognition Domain) support via py-ard

- **Production-Ready Infrastructure**
  - Smart embedding caching system
  - Batch processing optimization
  - GPU acceleration with automatic detection
  - CPU fallback for environments without GPU

- **Advanced Analysis Tools**
  - Donor-recipient matching analysis
  - Similarity metrics and visualization
  - t-SNE, UMAP, and PCA plots
  - Publication-ready figure generation

- **Comprehensive Documentation**
  - API reference
  - Installation guide
  - Troubleshooting guide
  - Example scripts and tutorials

### Technical Details

- Python 3.9+ required
- MIT License
- Fully typed with type hints
- Google-style docstrings throughout
- Comprehensive test suite

---

## [0.2.0] - 2025-01-01 (Pre-release)

### Added
- ProtT5 encoder integration
- Ankh encoder (base and large variants)
- Multi-encoder comparison example
- Comprehensive benchmark suite
- Biological correctness tests
- Architecture Decision Record for hybrid embeddings

### Changed
- Updated transformers requirement to >=4.30.0
- Improved batch encoding performance
- Enhanced documentation

### Fixed
- Added missing LICENSE file
- Improved error handling in encoders

---

## [0.1.0] - 2024-11-01 (Pre-release)

### Added
- Initial implementation with ProtBERT and ESM-2 encoders
- IMGT/HLA database downloader and parser
- Basic embedding generation pipeline
- Locus-specific analysis tools
- Example scripts for common use cases
