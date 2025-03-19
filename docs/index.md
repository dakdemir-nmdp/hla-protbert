# HLA-ProtBERT Documentation

Welcome to the HLA-ProtBERT documentation. This comprehensive framework enables encoding HLA alleles using ProtBERT and applying these embeddings to clinical prediction tasks.

## Getting Started

If you're new to HLA-ProtBERT, we recommend starting with:

- [Installation Guide](../README.md#installation) - Instructions for setting up HLA-ProtBERT
- [Quick Start Guide](../README.md#quick-start) - Basic usage examples
- [Getting Started Tutorial](tutorials/getting_started.md) - A step-by-step introduction to HLA-ProtBERT

## API Documentation

Detailed documentation for HLA-ProtBERT's key modules:

### Core Components

- [HLAEncoder](api/encoder.md) - Base class for HLA sequence encoders with common functionality
- [ProtBERTEncoder](api/protbert.md) - Implementation of HLA encoder using ProtBERT

### Data Handling
- IMGT/HLA Downloader - Tools for downloading and processing the IMGT/HLA database
- Sequence Utilities - Utilities for processing and analyzing HLA sequences

### Analysis Tools
- Matching Analysis - Tools for donor-recipient matching
- Visualization - Tools for visualizing HLA embeddings and similarity

## Tutorials

Step-by-step guides to help you accomplish common tasks:

- [Getting Started with HLA-ProtBERT](tutorials/getting_started.md) - Basic usage and concepts
- More tutorials coming soon:
  - Data Processing Tutorial
  - Clinical Prediction Tutorial
  - Advanced Analysis Tutorial
  - Fine-tuning ProtBERT for Custom Tasks

## Examples

The `examples/` directory contains practical examples demonstrating key functionality:

- [Basic Encoding](../examples/basic_encoding.py) - Basic HLA allele encoding
- [Donor Matching](../examples/donor_matching.py) - Donor-recipient HLA matching analysis
- [Clinical Prediction](../examples/clinical_prediction.py) - Clinical outcome prediction

## Additional Resources

- [GitHub Repository](https://github.com/dakdemir-nmdp/hla-protbert) - Source code and issues
- [IMGT/HLA Database](https://www.ebi.ac.uk/ipd/imgt/hla/) - Source of HLA sequence data
- [ProtBERT](https://github.com/agemagician/ProtTrans) - Pre-trained protein language models
