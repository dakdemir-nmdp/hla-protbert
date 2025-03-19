#!/usr/bin/env python3
"""
This script encodes FASTA files for all loci using the ProtBERT encoder,
computes t-SNE and UMAP projections for the embeddings, and generates scatter plots
colored and labeled by serological allele codes.

Dependencies:
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - umap-learn
  - (Project-specific) src.models.encoder: must provide an encode_fasta function.

Usage:
  python scripts/encode_sequences.py --data-dir data/raw --output-dir data/processed
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import umap
import logging

# Add parent directory to path for module imports
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

# Import project modules
from src.models.protbert import ProtBERTEncoder
from src.utils.logging import setup_logging

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

def create_sequences_pickle(fasta_file, output_file):
    """Create a pickle file with sequences from FASTA file"""
    headers, sequences = parse_fasta(fasta_file)
    
    # Create a dictionary mapping allele names to sequences
    sequence_dict = {}
    for i, header in enumerate(headers):
        # Extract allele name (assuming format like "HLA-A*01:01")
        # Some headers might have additional info after spaces
        allele = header.split()[0]
        if allele.startswith("HLA-"):
            allele = allele[4:]  # Remove HLA- prefix
        sequence_dict[allele] = sequences[i]
    
    # Save as pickle
    with open(output_file, 'wb') as f:
        pickle.dump(sequence_dict, f)
    
    return sequence_dict

def generate_embeddings(sequence_dict, output_dir, locus=None):
    """Generate embeddings for sequences using ProtBERT"""
    logger = logging.getLogger(__name__)
    sequences_file = os.path.join(output_dir, "hla_sequences.pkl")
    
    # Save sequences to pickle if not already done
    if not os.path.exists(sequences_file):
        logger.info(f"Saving sequences to {sequences_file}")
        with open(sequences_file, 'wb') as f:
            pickle.dump(sequence_dict, f)
    
    # Create embeddings directory
    embeddings_dir = os.path.join(os.path.dirname(output_dir), "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    try:
        # Initialize encoder
        logger.info("Initializing ProtBERT encoder...")
        encoder = ProtBERTEncoder(
            sequence_file=sequences_file,
            cache_dir=embeddings_dir,
            locus=locus
        )
        
        # Get all alleles for the locus or all alleles if locus is None
        if locus:
            alleles = [allele for allele in sequence_dict.keys() if allele.startswith(f"{locus}*")]
        else:
            alleles = list(sequence_dict.keys())
        
        # Encode alleles in batches
        logger.info(f"Encoding {len(alleles)} alleles...")
        embeddings = encoder.batch_encode_alleles(alleles)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def generate_visualizations(embeddings, output_dir, prefix="all"):
    """Generate t-SNE and UMAP visualizations for embeddings"""
    logger = logging.getLogger(__name__)
    
    # Extract data
    alleles = list(embeddings.keys())
    embedding_matrix = np.stack(list(embeddings.values()))
    
    # Extract locus from allele names for coloring
    loci = [allele.split('*')[0] for allele in alleles]
    
    # Create plot directory
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Generate t-SNE projection
    logger.info("Computing t-SNE projection...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding_matrix)-1))
        tsne_result = tsne.fit_transform(embedding_matrix)
        
        # Plot t-SNE
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            x=tsne_result[:, 0], 
            y=tsne_result[:, 1],
            hue=loci,
            palette="tab10",
            s=100,
            alpha=0.7
        )
        
        # Add labels for a subset of points (avoid overcrowding)
        if len(alleles) < 50:
            for i, allele in enumerate(alleles):
                plt.text(tsne_result[i, 0], tsne_result[i, 1], allele, fontsize=8)
        
        plt.title(f"t-SNE Projection of HLA Embeddings ({prefix})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{prefix}_tsne.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating t-SNE: {e}")
    
    # Generate UMAP projection
    logger.info("Computing UMAP projection...")
    try:
        umap_model = umap.UMAP(n_neighbors=min(15, len(embedding_matrix)-1), min_dist=0.1, random_state=42)
        umap_result = umap_model.fit_transform(embedding_matrix)
        
        # Plot UMAP
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            x=umap_result[:, 0], 
            y=umap_result[:, 1],
            hue=loci,
            palette="tab10",
            s=100,
            alpha=0.7
        )
        
        # Add labels for a subset of points
        if len(alleles) < 50:
            for i, allele in enumerate(alleles):
                plt.text(umap_result[i, 0], umap_result[i, 1], allele, fontsize=8)
        
        plt.title(f"UMAP Projection of HLA Embeddings ({prefix})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{prefix}_umap.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating UMAP: {e}")

def main():
    parser = argparse.ArgumentParser(description="Encode HLA sequences and generate visualizations.")
    parser.add_argument('--data-dir', default='data/raw', 
                        help='Directory containing FASTA files')
    parser.add_argument('--output-dir', default='data/processed', 
                        help='Directory to save outputs')
    parser.add_argument('--locus', 
                        help='Specific HLA locus to process (e.g., A, B, C)')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process consolidated file
    consolidated_file = os.path.join(args.data_dir, "hla_prot.fasta")
    if not os.path.exists(consolidated_file):
        logger.error(f"Consolidated file not found: {consolidated_file}")
        logger.error("Please run scripts/download_imgt_data.py first.")
        return 1
    
    # Parse FASTA and create pickle
    logger.info(f"Processing {consolidated_file}...")
    sequences_file = os.path.join(args.output_dir, "hla_sequences.pkl")
    
    if os.path.exists(sequences_file):
        logger.info(f"Loading existing sequences from {sequences_file}")
        with open(sequences_file, 'rb') as f:
            sequence_dict = pickle.load(f)
    else:
        logger.info(f"Parsing sequences and creating {sequences_file}")
        sequence_dict = create_sequences_pickle(consolidated_file, sequences_file)
    
    # Print report
    logger.info(f"Total sequences: {len(sequence_dict)}")
    
    # Analyze sequence lengths
    lengths = [len(seq) for seq in sequence_dict.values()]
    logger.info(f"Sequence length statistics:")
    logger.info(f"  Minimum: {min(lengths)}")
    logger.info(f"  Maximum: {max(lengths)}")
    logger.info(f"  Average: {sum(lengths)/len(lengths):.1f}")
    
    # Count alleles per locus
    locus_counts = {}
    for allele in sequence_dict.keys():
        locus = allele.split('*')[0] if '*' in allele else 'unknown'
        locus_counts[locus] = locus_counts.get(locus, 0) + 1
    
    logger.info(f"Alleles per locus:")
    for locus, count in sorted(locus_counts.items()):
        logger.info(f"  {locus}: {count}")
    
    # Generate embeddings for specific locus or all
    try:
        if args.locus:
            logger.info(f"Generating embeddings for locus {args.locus}...")
            embeddings = generate_embeddings(sequence_dict, str(output_dir), locus=args.locus)
            prefix = args.locus
        else:
            logger.info("Generating embeddings for all loci...")
            embeddings = generate_embeddings(sequence_dict, str(output_dir))
            prefix = "all"
        
        # Generate visualizations
        if not args.skip_visualizations and embeddings:
            logger.info("Generating visualizations...")
            generate_visualizations(embeddings, str(output_dir), prefix=prefix)
        
    except Exception as e:
        logger.error(f"Error in encoding process: {e}")
        return 1
    
    logger.info("Encoding completed successfully!")
    return 0

if __name__ == '__main__':
    exit(main())
