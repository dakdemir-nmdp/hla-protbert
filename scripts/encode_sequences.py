#!/usr/bin/env python3
"""
This script encodes FASTA files using a specified encoder (ProtBERT or ESM3),
computes t-SNE and UMAP projections for the embeddings, and generates scatter plots.

Dependencies:
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - umap-learn
  - (Project-specific) src.models.encoder: must provide an encode_fasta function.

Usage:
  python scripts/encode_sequences.py --encoder-type [protbert|esm3] --data-dir data/raw --output-dir data/processed
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
from src.models.encoders import ProtBERTEncoder, ESMEncoder # Updated import
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager # Import ConfigManager if needed for defaults

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

def generate_embeddings(
    sequence_dict,
    base_output_dir, # Renamed from output_dir
    encoder_type,
    model_name,
    device,
    batch_size,
    locus=None,
    verify_ssl=True # Add verify_ssl parameter
):
    """Generate embeddings for sequences using the specified encoder"""
    logger = logging.getLogger(__name__)
    sequences_file = os.path.join(base_output_dir, "hla_sequences.pkl") # Sequences are shared

    # Save sequences to pickle if not already done (remains in base processed dir)
    if not os.path.exists(sequences_file):
        logger.info(f"Saving sequences to {sequences_file}")
        with open(sequences_file, 'wb') as f:
            pickle.dump(sequence_dict, f)

    # Determine encoder class and specific settings
    if encoder_type == "protbert":
        EncoderClass = ProtBERTEncoder
        cache_subdir = "protbert"
        # Add any specific ProtBERT args if needed from config or defaults
        encoder_specific_args = {
            "use_peptide_binding_region": True, # Example default
            # "verify_ssl": verify_ssl # ProtBERT might not need this either if using transformers
        }
    elif encoder_type == "esm": # Changed 'esm3' to 'esm'
        EncoderClass = ESMEncoder # Updated class name
        cache_subdir = "esm" # Updated cache subdir
        encoder_specific_args = {
            # verify_ssl is no longer needed by ESMEncoder
        }
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    # Define encoder-specific cache directory
    # Assumes base_output_dir is like 'data/processed', goes up one level for 'data/embeddings'
    base_embeddings_dir = Path(base_output_dir).parent / "embeddings"
    final_cache_dir = base_embeddings_dir / cache_subdir
    final_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the selected encoder
        logger.info(f"Initializing {encoder_type.upper()} encoder (model={model_name}, device={device})...")
        encoder = EncoderClass(
            sequence_file=sequences_file,
            cache_dir=final_cache_dir,
            model_name=model_name,
            locus=locus,
            device=device,
            **encoder_specific_args # Pass any specific args
        )

        # Get all alleles for the locus or all alleles if locus is None
        if locus:
            alleles = [allele for allele in sequence_dict.keys() if allele.startswith(f"{locus}*")]
        else:
            alleles = list(sequence_dict.keys())

        # Encode alleles in batches using the encoder's method
        logger.info(f"Encoding {len(alleles)} alleles with {encoder_type.upper()}...")
        # Use the batch_size parameter
        embeddings = encoder.batch_encode_alleles(alleles, batch_size=batch_size)

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def generate_visualizations(embeddings, base_output_dir, encoder_type, prefix="all"):
    """Generate t-SNE and UMAP visualizations for embeddings"""
    logger = logging.getLogger(__name__)

    if not embeddings:
        logger.warning("No embeddings provided for visualization.")
        return

    # Extract data
    alleles = list(embeddings.keys())
    embedding_matrix = np.stack(list(embeddings.values()))
    
    # Extract locus from allele names for coloring
    loci = [allele.split('*')[0] for allele in alleles]
    
    # Create encoder-specific plot directory
    plot_dir = Path(base_output_dir) / encoder_type / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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

        plt.title(f"t-SNE Projection of HLA Embeddings ({encoder_type.upper()} - {prefix})")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_tsne.png", dpi=300)
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

        plt.title(f"UMAP Projection of HLA Embeddings ({encoder_type.upper()} - {prefix})")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_umap.png", dpi=300)
        plt.close()

    except Exception as e:
        logger.error(f"Error generating UMAP: {e}")

def main():
    parser = argparse.ArgumentParser(description="Encode HLA sequences using a specified encoder and generate visualizations.")
    parser.add_argument('--encoder-type', choices=["protbert", "esm"], default="protbert", # Changed 'esm3' to 'esm'
                        help="Type of encoder model to use (default: protbert)")
    parser.add_argument('--data-dir', default='data/raw',
                        help='Directory containing FASTA files')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Base directory to save outputs (sequences pkl, encoder-specific plots)')
    parser.add_argument('--locus',
                        help='Specific HLA locus to process (e.g., A, B, C)')
    parser.add_argument('--model', help="Model name or path for the selected encoder (optional, uses defaults)")
    parser.add_argument('--device', choices=["cpu", "cuda"], help="Device to run model on (optional, uses auto-detect)")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for encoding (default: 8)")
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--no-verify-ssl', action='store_false', dest='verify_ssl',
                        help='Disable SSL certificate verification for model downloads (use with caution)')
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    config = ConfigManager() # Load default config if needed for model names etc.

    # Ensure base output directory exists
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)

    # Determine model name, device
    device = args.device or config.get("encoder.default_device", "auto")
    if args.encoder_type == "protbert":
        default_model = "Rostlab/prot_bert"
        model_config_key = "model.protbert_model"
    else: # esm
        default_model = "facebook/esm2_t33_650M_UR50D" # Updated default ESM model
        model_config_key = "model.esm_model" # Updated config key
    model_name = args.model or config.get(model_config_key, default_model)


    # Process consolidated file (remains in base data dir)
    consolidated_file = Path(args.data_dir) / "hla_prot.fasta"
    if not os.path.exists(consolidated_file):
        logger.error(f"Consolidated file not found: {consolidated_file}")
        logger.error("Please run scripts/download_imgt_data.py first.")
        return 1
    
    # Parse FASTA and create pickle (saved in base output dir)
    logger.info(f"Processing {consolidated_file}...")
    sequences_file = base_output_dir / "hla_sequences.pkl"

    if sequences_file.exists():
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
    
    # Generate embeddings for specific locus or all using the selected encoder
    try:
        prefix = args.locus if args.locus else "all"
        logger.info(f"Generating {args.encoder_type.upper()} embeddings for: {prefix}")

        embeddings = generate_embeddings(
            sequence_dict=sequence_dict,
            base_output_dir=str(base_output_dir),
            encoder_type=args.encoder_type,
            model_name=model_name,
            device=device,
            batch_size=args.batch_size,
            locus=args.locus,
            # verify_ssl=args.verify_ssl # Removed verify_ssl argument for ESMEncoder
        )

        # Generate visualizations (saved in encoder-specific subdir)
        if not args.skip_visualizations and embeddings:
            logger.info(f"Generating visualizations for {args.encoder_type.upper()} embeddings...")
            generate_visualizations(
                embeddings=embeddings,
                base_output_dir=str(base_output_dir),
                encoder_type=args.encoder_type,
                prefix=prefix
            )

    except Exception as e:
        logger.error(f"Error in {args.encoder_type.upper()} encoding/visualization process: {e}")
        return 1

    logger.info(f"{args.encoder_type.upper()} encoding completed successfully!")
    return 0

if __name__ == '__main__':
    exit(main())
