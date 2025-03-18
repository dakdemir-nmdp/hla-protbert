#!/usr/bin/env python3
"""
This script downloads data, encodes FASTA files for all loci using the provided encoder,
computes t-SNE and UMAP projections for the embeddings, and generates scatter plots
colored and labeled by serological allele codes.

Dependencies:
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - umap-learn
  - (Project-specific) src.models.encoder: must provide an encode_fasta function.
  - (Optional) src.data.imgt_downloader: if download_data function is available for data download.

Usage:
  python scripts/download_and_encode.py --data-dir data/raw --output-dir data/processed
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

# Import the encoder function from the project module.
# This function should take a FASTA file path as input and return a numpy array embedding.
from src.models.encoder import encode_fasta

# Optional: Uncomment the following line if a download_data function exists
# from src.data.imgt_downloader import download_data

def extract_allele_code(allele_name):
    """
    Extract the serological allele code from a full allele name.
    For example: "DRB1*01:02" becomes "DRB1*01".
    """
    return allele_name.split(':')[0]

def main():
    parser = argparse.ArgumentParser(description="Download data, encode FASTA files, compute projections, and generate plots.")
    parser.add_argument('--data-dir', default='data/raw', help='Directory containing FASTA files')
    parser.add_argument('--output-dir', default='data/processed', help='Directory to save outputs')
    args = parser.parse_args()
    
    # Ensure output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optional: Download data if necessary.
    # download_data()

    # Find all FASTA files in the data directory.
    fasta_files = glob.glob(os.path.join(args.data_dir, '*.fasta'))
    if not fasta_files:
        print(f"No FASTA files found in {args.data_dir}")
        return
    
    embeddings = []
    allele_labels = []
    
    for fasta_file in fasta_files:
        # Extract allele code from filename.
        # Assumes the FASTA filename is in a format like "DRB1*01:02.fasta"
        base_name = os.path.basename(fasta_file)
        allele_full = os.path.splitext(base_name)[0]
        allele_code = extract_allele_code(allele_full)
        
        # Generate embedding using the encoder.
        embedding = encode_fasta(fasta_file)
        embeddings.append(embedding)
        allele_labels.append(allele_code)
        print(f"Processed {fasta_file} with allele code {allele_code}")
    
    embeddings = np.array(embeddings)
    
    # Compute t-SNE projection.
    tsne = TSNE(n_components=2, random_state=42)
    tsne_proj = tsne.fit_transform(embeddings)
    
    # Compute UMAP projection.
    reducer = umap.UMAP(random_state=42)
    umap_proj = reducer.fit_transform(embeddings)
    
    # Plot t-SNE projection.
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_proj[:, 0], y=tsne_proj[:, 1], hue=allele_labels, palette='deep')
    plt.title("t-SNE Projection")
    plt.legend(title="Allele Code")
    tsne_fig_path = os.path.join(args.output_dir, 'tsne_projection.png')
    plt.savefig(tsne_fig_path)
    plt.close()
    print(f"Saved t-SNE plot to {tsne_fig_path}")
    
    # Plot UMAP projection.
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_proj[:, 0], y=umap_proj[:, 1], hue=allele_labels, palette='deep')
    plt.title("UMAP Projection")
    plt.legend(title="Allele Code")
    umap_fig_path = os.path.join(args.output_dir, 'umap_projection.png')
    plt.savefig(umap_fig_path)
    plt.close()
    print(f"Saved UMAP plot to {umap_fig_path}")
    
    # Save embeddings and projections for further analysis.
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'tsne_projection.npy'), tsne_proj)
    np.save(os.path.join(args.output_dir, 'umap_projection.npy'), umap_proj)
    print("Saved embeddings and projection arrays.")

if __name__ == '__main__':
    main()
