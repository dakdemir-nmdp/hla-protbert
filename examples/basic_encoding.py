#!/usr/bin/env python
"""
Basic HLA Encoding Example
------------------------
Demonstrates how to use the HLA-ProtBERT system to encode HLA alleles.
"""
import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.models.encoders import ProtBERTEncoder, ESMEncoder # Updated import
from src.models.encoder import HLAEncoder # Import base class for type hinting
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def main():
    """Main function demonstrating basic HLA encoding"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Encode HLA alleles with selected encoder")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--alleles", required=True, nargs='+', help="HLA alleles to encode (space separated)")
    parser.add_argument("--locus", help="Specific HLA locus to use for encoder")
    parser.add_argument("--find-similar", dest="find_similar", action="store_true", help="Find similar alleles")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of similar alleles to find")
    parser.add_argument("--compare", action="store_true", help="Compare similarity between provided alleles")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Add encoder selection arguments
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["protbert", "esm"], # Changed 'esm3' to 'esm'
        default="protbert",
        help="Type of encoder model to use ('protbert' or 'esm')" # Updated help text
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None, # Default handled later based on encoder type
        help="Specific model name/path (e.g., 'Rostlab/prot_bert' or 'facebook/esm2_t33_650M_UR50D')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on ('cpu', 'cuda', 'auto')"
    )
    parser.add_argument(
        "--pooling-strategy",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy for embeddings ('mean' or 'cls') - primarily for ESM" # Updated help text
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_false", # Sets verify_ssl to False when flag is present
        dest="verify_ssl",
        default=True, # Default to verifying SSL
        help="Disable SSL certificate verification for model downloads"
    )

    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Determine paths
    data_dir = args.data_dir or config.get("data.base_dir", str(project_dir / "data"))
    embeddings_base_dir = config.get("data.embeddings_dir", os.path.join(data_dir, "embeddings"))
    
    # Create encoder-specific cache directory
    if args.encoder_type == "protbert":
        cache_subdir = "protbert"
    elif args.encoder_type == "esm":
        cache_subdir = "esm"
    else:
        cache_subdir = "default"
    
    embeddings_dir = os.path.join(embeddings_base_dir, cache_subdir)
    sequences_file = config.get("data.sequences_file", os.path.join(data_dir, "processed", "hla_sequences.pkl"))
    
    # Initialize encoder based on type
    logger.info(f"Initializing {args.encoder_type.upper()} encoder (locus={args.locus or 'all'})...")
    try:
        encoder_args = {
            "sequence_file": sequences_file,
            "cache_dir": embeddings_dir,
            "locus": args.locus,
            "device": args.device,
            # Add model_name if provided, otherwise let class use default
            **({"model_name": args.model_name} if args.model_name else {})
        }

        if args.encoder_type == "protbert":
            EncoderClass = ProtBERTEncoder
            # ProtBERT specific args
            encoder_args["pooling_strategy"] = args.pooling_strategy # ProtBERT also uses this
            encoder_args["use_peptide_binding_region"] = config.get("model.use_peptide_binding_region", True)
            if not args.model_name:
                 encoder_args["model_name"] = config.get("model.protbert_model_name", "Rostlab/prot_bert")
        elif args.encoder_type == "esm": # Changed 'esm3' to 'esm'
            EncoderClass = ESMEncoder # Updated class name
            # ESM specific args
            encoder_args["pooling_strategy"] = args.pooling_strategy
            # verify_ssl is no longer needed by ESMEncoder
            # Add HF token from config if available
            hf_token = config.get("model.hf_token", None)
            if hf_token:
                encoder_args["hf_token"] = hf_token
            if not args.model_name:
                 # Use the new default from ESMEncoder class
                 encoder_args["model_name"] = config.get("model.esm_model_name", "facebook/esm2_t33_650M_UR50D")
        else:
            # This case should not be reached due to argparse choices
            raise ValueError(f"Unsupported encoder type: {args.encoder_type}")

        encoder: HLAEncoder = EncoderClass(**encoder_args) # Add type hint

    except FileNotFoundError:
        logger.error(f"Sequence file not found: {sequences_file}")
        logger.error("Please run scripts/update_imgt.py and scripts/generate_embeddings.py first.")
        return
    
    # Encode alleles
    print("\n===== HLA Embedding Example =====\n")
    print(f"Using {args.encoder_type.upper()} model: {encoder.model_name}")
    print(f"Pooling strategy: {encoder.pooling_strategy}")
    if args.encoder_type == "protbert":
        print(f"Using peptide binding region: {encoder.use_peptide_binding_region}")
    print(f"Device: {encoder.device}")
    
    # Get embeddings for each allele
    embeddings = {}
    for allele in args.alleles:
        try:
            print(f"\nEncoding {allele}...")
            
            # Get protein sequence
            sequence = encoder.get_sequence(allele)
            if sequence is None:
                print(f"  Warning: No sequence found for {allele}")
                continue
                
            print(f"  Sequence: {sequence[:50]}..." if len(sequence) > 50 else f"  Sequence: {sequence}")
            print(f"  Sequence length: {len(sequence)} amino acids")
            
            # Get embedding
            embedding = encoder.get_embedding(allele)
            embeddings[allele] = embedding
            
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
            
        except Exception as e:
            logger.error(f"Error encoding {allele}: {e}")
    
    # Find similar alleles if requested
    if args.find_similar and embeddings:
        print("\n----- Finding Similar Alleles -----\n")
        
        for allele in args.alleles:
            if allele not in embeddings:
                continue
                
            print(f"Alleles similar to {allele}:")
            similar = encoder.find_similar_alleles(allele, top_k=args.top_k)
            
            if not similar:
                print("  No similar alleles found")
                continue
                
            for similar_allele, similarity in similar:
                print(f"  {similar_allele}: similarity={similarity:.4f}")
    
    # Compare alleles if requested
    if args.compare and len(args.alleles) > 1 and len(embeddings) > 1:
        print("\n----- Comparing Alleles -----\n")
        
        # Calculate pairwise similarities
        print("Pairwise similarities:")
        
        for i, allele1 in enumerate(args.alleles):
            if allele1 not in embeddings:
                continue
                
            for allele2 in args.alleles[i+1:]:
                if allele2 not in embeddings:
                    continue
                    
                # Calculate cosine similarity
                similarity = encoder._cosine_similarity(embeddings[allele1], embeddings[allele2])
                print(f"  {allele1} vs {allele2}: {similarity:.4f}")
                
                # Check if they represent the same protein
                if similarity > 0.99:
                    print(f"    These alleles are identical or encode the same protein")
                elif similarity > 0.95:
                    print(f"    These alleles are very similar (likely same protein group)")
                elif similarity > 0.90:
                    print(f"    These alleles are functionally similar")
                elif similarity < 0.70:
                    print(f"    These alleles are substantially different")

if __name__ == "__main__":
    main()
