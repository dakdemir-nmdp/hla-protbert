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

def parse_fasta(fasta_file):
    """Parse a FASTA file and return sequences and headers"""
    sequences = []
    headers = []
    current_seq = []
    current_header = None
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:  # Save the previous sequence
                    sequences.append(''.join(current_seq))
                    current_seq = []
                current_header = line[1:]  # Remove '>' from header
                headers.append(current_header)
            else:
                current_seq.append(line)
        if current_seq:  # Save the last sequence
            sequences.append(''.join(current_seq))
    
    return headers, sequences

def main():
    parser = argparse.ArgumentParser(description="Process HLA sequences.")
    parser.add_argument('--data-dir', default='data/raw', help='Directory containing FASTA files')
    parser.add_argument('--output-dir', default='data/processed', help='Directory to save outputs')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the consolidated file
    consolidated_file = os.path.join(args.data_dir, "hla_prot.fasta")
    if not os.path.exists(consolidated_file):
        print(f"Consolidated file not found: {consolidated_file}")
        return

    print(f"Processing {consolidated_file}...")
    headers, sequences = parse_fasta(consolidated_file)
    
    # Print report
    print(f"\nTotal sequences: {len(sequences)}")
    
    # Analyze sequence lengths
    lengths = [len(seq) for seq in sequences]
    print("\nSequence length statistics:")
    print(f"  Minimum: {min(lengths)}")
    print(f"  Maximum: {max(lengths)}")
    print(f"  Average: {sum(lengths)/len(lengths):.1f}")
    
    # Count alleles per locus
    locus_counts = {}
    for header in headers:
        locus = header.split('*')[0]
        locus_counts[locus] = locus_counts.get(locus, 0) + 1
    
    print("\nAlleles per locus:")
    for locus, count in sorted(locus_counts.items()):
        print(f"  {locus}: {count}")

if __name__ == '__main__':
    main()
