#!/usr/bin/env python
"""
Generate HLA Embeddings
---------------------
Script to generate and cache embeddings for HLA alleles using a specified encoder.
"""
import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.models.encoders import ProtBERTEncoder, ESMEncoder
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def load_alleles_from_file(file_path: Path) -> List[str]:
    """Load alleles from file
    
    Args:
        file_path: Path to file with alleles
        
    Returns:
        List of allele names
    """
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    # Determine file type from extension
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(file_path)
        # Try to find column with HLA alleles
        allele_col = None
        for col in df.columns:
            if col.lower() in ['allele', 'hla', 'hla_allele', 'allele_name']:
                allele_col = col
                break
        
        if allele_col is None:
            # Assume first column contains alleles
            allele_col = df.columns[0]
            
        alleles = df[allele_col].tolist()
        
    elif suffix == '.txt':
        # Assume text file with one allele per line
        with open(file_path, 'r') as f:
            alleles = [line.strip() for line in f if line.strip()]
    
    elif suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
        # Try to find column with HLA alleles
        allele_col = None
        for col in df.columns:
            if col.lower() in ['allele', 'hla', 'hla_allele', 'allele_name']:
                allele_col = col
                break
        
        if allele_col is None:
            # Assume first column contains alleles
            allele_col = df.columns[0]
            
        alleles = df[allele_col].tolist()
    
    else:
        logger.error(f"Unsupported file format: {suffix}")
        return []
    
    # Standardize allele names
    standardized = []
    for allele in alleles:
        if allele:
            standardized.append(str(allele).strip())
    
    return standardized

def main():
    """Main function to generate embeddings"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate HLA embeddings using a specified encoder.")
    parser.add_argument("--encoder-type", choices=["protbert", "esm"], default="protbert",
                        help="Type of encoder model to use (default: protbert)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--sequences", help="Path to HLA sequences pickle file")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Base directory to cache embeddings (will create encoder-specific subdirs)")
    parser.add_argument("--allele-file", dest="allele_file", help="File with alleles to encode (CSV or TXT)")
    parser.add_argument("--locus", help="Generate embeddings for specific locus only")
    parser.add_argument("--model", help="Model name or path for the selected encoder")
    parser.add_argument("--all", action="store_true", help="Generate embeddings for all known alleles")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run model on")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size for encoding")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing embeddings")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Determine paths and parameters
    data_dir = args.data_dir or config.get("data.base_dir", str(project_dir / "data"))
    sequences_file = args.sequences or config.get(
        "data.sequences_file", 
        os.path.join(data_dir, "processed", "hla_sequences.pkl")
    )
    cache_dir = args.cache_dir or config.get(
        "data.embeddings_dir",
        os.path.join(data_dir, "embeddings") # Base embeddings dir
    )
    device = args.device or config.get("encoder.default_device", "auto")
    batch_size = args.batch_size or config.get("model.batch_size", 8)

    # Determine model name and cache dir based on encoder type
    if args.encoder_type == "protbert":
        EncoderClass = ProtBERTEncoder
        default_model = "Rostlab/prot_bert"
        model_config_key = "model.protbert_model"
        cache_subdir = "protbert"
        pooling_strategy = config.get("model.pooling_strategy", "mean") # ProtBERT specific
        use_pbr = config.get("model.use_peptide_binding_region", True) # ProtBERT specific
    elif args.encoder_type == "esm":
        EncoderClass = ESMEncoder
        default_model = "facebook/esm2_t33_650M_UR50D"
        model_config_key = "model.esm_model"
        cache_subdir = "esm"
        pooling_strategy = config.get("model.esm_pooling_strategy", "mean")
        use_pbr = False
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        return 1

    model_name = args.model or config.get(model_config_key, default_model)
    final_cache_dir = Path(cache_dir) / cache_subdir # Specific cache dir for this encoder

    # Initialize encoder
    logger.info(f"Initializing {args.encoder_type.upper()} encoder (model={model_name}, device={device})")
    try:
        encoder_args = {
            "sequence_file": sequences_file,
            "cache_dir": final_cache_dir,
            "model_name": model_name,
            "locus": args.locus,
            "device": device,
            "pooling_strategy": pooling_strategy
        }
        
        # Add HF token if available in config (for ESM)
        if args.encoder_type == "esm":
            hf_token = config.get("model.hf_token", None)
            if hf_token:
                encoder_args["hf_token"] = hf_token
                
        # Add ProtBERT specific args if applicable
        if args.encoder_type == "protbert":
            encoder_args["use_peptide_binding_region"] = use_pbr
            # Potentially add verify_ssl for ProtBERT if it still needs it

        encoder = EncoderClass(**encoder_args)

    except Exception as e:
        logger.error(f"Failed to initialize {args.encoder_type.upper()} encoder: {e}")
        return 1

    # Determine which alleles to encode
    alleles = []
    
    if args.all:
        # Get all alleles from encoder's sequences
        if args.locus:
            alleles = [
                allele for allele in encoder.sequences.keys()
                if allele.startswith(f"{args.locus}*")
            ]
            logger.info(f"Found {len(alleles)} {args.locus} alleles to encode")
        else:
            alleles = list(encoder.sequences.keys())
            logger.info(f"Found {len(alleles)} total alleles to encode")
            
    elif args.allele_file:
        # Load alleles from file
        allele_file_path = Path(args.allele_file)
        alleles = load_alleles_from_file(allele_file_path)
        logger.info(f"Loaded {len(alleles)} alleles from {allele_file_path}")
        
        # Filter by locus if specified
        if args.locus:
            alleles = [
                allele for allele in alleles
                if allele.startswith(f"{args.locus}*")
            ]
            logger.info(f"Filtered to {len(alleles)} {args.locus} alleles")
    
    elif args.locus:
        # Encode all alleles for specified locus
        alleles = [
            allele for allele in encoder.sequences.keys()
            if allele.startswith(f"{args.locus}*")
        ]
        logger.info(f"Found {len(alleles)} {args.locus} alleles to encode")
    
    else:
        logger.error("No alleles specified. Use --all, --locus, or --allele-file.")
        return 1
    
    # Skip alleles that are already cached unless forced
    if not args.force:
        alleles_to_encode = [
            allele for allele in alleles
            if allele not in encoder.embeddings
        ]
        
        if len(alleles_to_encode) < len(alleles):
            logger.info(f"Skipping {len(alleles) - len(alleles_to_encode)} already cached alleles")
            alleles = alleles_to_encode
    
    if not alleles:
        logger.info("No new alleles to encode.")
        return 0 # Return 0 if no alleles to encode
    
    # Generate embeddings
    logger.info(f"Generating {args.encoder_type.upper()} embeddings for {len(alleles)} alleles in batches of {batch_size}...")

    # Use batch encoding for better performance
    logger.info(f"Performing batch encoding with {args.encoder_type.upper()}...")
    # The batch_encode_alleles method is overridden in ESMEncoder to use its specific batch_encode
    # For ProtBERT, it uses the base class implementation which calls its specific batch_encode.
    # No change needed here as the encoder object handles the correct batching.

    try:
        # Call the batch encode method on the instantiated encoder
        # This method handles sequence retrieval, encoding, caching internally
        # Pass the force argument
        encoder.batch_encode_alleles(alleles, batch_size=batch_size, force=args.force)

    except Exception as e:
        logger.error(f"Error during {args.encoder_type.upper()} batch encoding: {e}")
        return 1

    # Save final cache (handled within batch_encode_alleles now)
    # logger.info(f"Saving final embeddings cache ({len(encoder.embeddings)} total embeddings)")
    # encoder._save_embedding_cache() # Already saved within batch_encode_alleles

    logger.info(f"{args.encoder_type.upper()} embedding generation complete")
    return 0 # Return 0 on success

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        # Catch any uncaught exceptions from main()
        logging.exception(f"Unhandled exception occurred: {e}")
        sys.exit(1)
