# HLA-ProtBERT Presentation

Project briefings and slides for the HLA-ProtBERT codebase live here. The slides now reflect the current multi-encoder pipeline (ProtBERT, ESM-2, ProtT5, Ankh Base, Ankh Large) and the latest IMGT/HLA sequence snapshot.

## Contents

- `hla_protbert_presentation_final.tex`: Main Beamer source (PDF committed)
- `references.bib`: Bibliography used by the deck
- `diagram_descriptions.txt`: Plain-language descriptions for the transformer and dimensionality reduction diagrams
- `transformer_diagram.pdf/png` and `dim_reduction_diagram.pdf/tex`: Supporting visuals
- `dimensionality_reduction.txt` and `transformer_architecture.txt`: Mermaid/flowchart sources for diagrams
- `ProtBertSummary.txt`: Speaker notes summarizing the project and encoders
- `data/`: Any presentation-specific exports (plots are pulled from the pipeline output)

## Image Sources

Slide images reference the locus analysis artifacts produced by the codebase:
```
data/analysis/locus_embeddings/class1/plots/
```

Current counts from `data/processed/hla_sequences.pkl` (Dec 2025 snapshot):
- Total alleles: 26,000
- Class I focus in slides: HLA-A 5,489; HLA-B 6,584; HLA-C 5,206 (17,279 combined)

## Regenerating Plots

Run the full pipeline to refresh embeddings and plots before re-compiling slides:
```bash
cd /Users/dakdemir/Library/CloudStorage/OneDrive-NMDP/Year2025/Github/hla-protbert
source venv/bin/activate
./run_complete_pipeline_all_encoders.sh [--disable-ssl-verify] [--ankh-backend auto|huggingface|ankh]
```

Key artifacts consumed by the slides:
- Sequence data: `data/processed/hla_sequences.pkl`
- ProtBERT embeddings (used for locus plots): `data/embeddings/protbert/hla_embeddings.pkl`
- Locus plots: `data/analysis/locus_embeddings/class1/plots/*.png`

## Build Instructions

You need a LaTeX distribution with Beamer and BibTeX.

```bash
cd /Users/dakdemir/Library/CloudStorage/OneDrive-NMDP/Year2025/Github/hla-protbert/presentations

pdflatex hla_protbert_presentation_final
bibtex hla_protbert_presentation_final
pdflatex hla_protbert_presentation_final
pdflatex hla_protbert_presentation_final
```

If using a LaTeX editor (TeXShop, TeXStudio, Overleaf):
1) Open `hla_protbert_presentation_final.tex`
2) Ensure `references.bib` is in the same directory
3) Build with BibTeX enabled, then re-run LaTeX to resolve citations

## Deck Focus

- HLA complexity and motivation
- Multi-encoder HLA-ProtBERT pipeline and caching
- Class I locus embeddings (A, B, C) with PCA/t-SNE/UMAP views
- Clinical, technical, and management takeaways
- Future work and extension paths (additional encoders, ensemble use, class II analyses)
