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

from src.models.protbert import ProtBERTEncoder
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def main():
    """Main function demonstrating basic HLA encoding"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Encode HLA alleles with ProtBERT")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--alleles", required=True, nargs='+', help="HLA alleles to encode (space separated)")
    parser.add_argument("--locus", help="Specific HLA locus to use for encoder")
    parser.add_argument("--find-similar", dest="find_similar", action="store_true", help="Find similar alleles")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of similar alleles to find")
    parser.add_argument("--compare", action="store_true", help="Compare similarity between provided alleles")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Determine paths
    data_dir = args.data_dir or config.get("data.base_dir", str(project_dir / "data"))
    embeddings_dir = config.get("data.embeddings_dir", os.path.join(data_dir, "embeddings"))
    sequences_file = config.get("data.sequences_file", os.path.join(data_dir, "processed", "hla_sequences.pkl"))
    
    # Initialize encoder
    logger.info(f"Initializing HLA encoder (locus={args.locus or 'all'})...")
    try:
        encoder = ProtBERTEncoder(
            sequence_file=sequences_file,
            cache_dir=embeddings_dir,
            locus=args.locus,
            pooling_strategy=config.get("model.pooling_strategy", "mean"),
            use_peptide_binding_region=config.get("model.use_peptide_binding_region", True)
        )
    except FileNotFoundError:
        logger.error(f"Sequence file not found: {sequences_file}")
        logger.error("Please run scripts/update_imgt.py and scripts/generate_embeddings.py first.")
        return
    
    # Encode alleles
    print("\n===== HLA Embedding Example =====\n")
    print(f"Using ProtBERT model: {encoder.model_name}")
    print(f"Pooling strategy: {encoder.pooling_strategy}")
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
