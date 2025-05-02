#\!/bin/bash
# Full HLA-ProtBERT data generation pipeline

set -e  # Exit on error

echo "Creating directory structure..."
mkdir -p data/raw/fasta data/processed data/embeddings/{protbert,esm} \
  data/analysis/locus_embeddings/{class1,class2}/{embeddings,plots,reports} \
  data/analysis/locus_embeddings/logs

echo "Downloading HLA sequence data..."
# Check the actual parameters expected by the script
python scripts/download_imgt_data.py --data-dir data/raw/fasta --verbose

echo "Processing downloaded data..."
# Adjust parameters to match actual script expectations 
python scripts/update_imgt.py --verbose

echo "Generating ProtBERT embeddings for all loci..."
python scripts/generate_embeddings.py --encoder-type protbert --all --verbose


echo "Copying embeddings to locus-specific directories..."
for locus in A B C; do
  cp data/embeddings/protbert/hla_embeddings.pkl data/analysis/locus_embeddings/class1/embeddings/hla_${locus}_embeddings.pkl
done

for locus in DRB1 DQB1 DPB1; do
  cp data/embeddings/protbert/hla_embeddings.pkl data/analysis/locus_embeddings/class2/embeddings/hla_${locus}_embeddings.pkl
done

echo "Running analysis for Class I loci..."
python scripts/run_locus_analysis.py --class1-only --debug

echo "Running analysis for Class II loci..."
python scripts/run_locus_analysis.py --class2-only --debug

echo "Running Jupyter notebooks..."
jupyter nbconvert --to notebook --execute notebooks/b_locus_clustering.ipynb --output executed_b_locus_clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/hla_protbert_demo.ipynb --output executed_hla_protbert_demo.ipynb
jupyter nbconvert --to notebook --execute notebooks/locus_embeddings_analysis.ipynb --output executed_locus_embeddings_analysis.ipynb

echo "Pipeline complete\! All data and visualizations are now generated."
