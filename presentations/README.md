# HLA-ProtBERT Presentation

This directory contains a LaTeX Beamer presentation about the HLA-ProtBERT model and its locus embedding analysis.

## Contents

- `hla_protbert_presentation.tex`: The main LaTeX source file for the presentation
- `references.bib`: Bibliography file with citations
- `diagram_descriptions.txt`: Descriptions of the diagrams referenced in the presentation

## Image References

The presentation references visualization images from the following directory:
```
data/analysis/locus_embeddings/class1/plots/
```

These images include:
- HLA-A, HLA-B, and HLA-C visualizations in PCA, t-SNE, and UMAP projections

## How to Build the Presentation

To build the presentation PDF, you need a LaTeX distribution installed with the Beamer package.

### Command Line

```bash
# Navigate to the presentations directory
cd presentations

# Compile the presentation (first pass)
pdflatex hla_protbert_presentation

# Compile the bibliography
bibtex hla_protbert_presentation

# Compile the presentation (second pass to include citations)
pdflatex hla_protbert_presentation

# Compile once more to resolve all references
pdflatex hla_protbert_presentation
```

### Using a LaTeX Editor

If you're using a LaTeX editor like TeXShop, TeXStudio, or Overleaf:

1. Open `hla_protbert_presentation.tex` in your editor
2. Make sure the bibliography file `references.bib` is in the same directory
3. Use the "Build and View" or equivalent option to compile the presentation

## Presentation Overview

This presentation covers:

1. Introduction to ProtBERT and its transformer architecture
2. Technical details of the ProtBERT model
3. Background on HLA (Human Leukocyte Antigens)
4. Implementation of HLA-ProtBERT
5. Locus embedding analysis methodology
6. Results of HLA Class I analysis (A, B, C loci)
7. Key findings and capabilities
8. Future research directions

The presentation is designed to be 8-10 slides in length, focusing on the Class I HLA analysis as requested.
