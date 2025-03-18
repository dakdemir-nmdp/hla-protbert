#!/usr/bin/env python
"""
Generate HLA Embeddings
---------------------
Script to generate and cache ProtBERT embeddings for HLA alleles.
"""
import os
import sys
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.models.protbert import ProtBERTEncoder
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
    parser = argparse.ArgumentParser(description="Generate ProtBERT embeddings for HLA alleles")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--sequences", help="Path to HLA sequences pickle file")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Directory to cache embeddings")
    parser.add_argument("--allele-file", dest="allele_file", help="File with alleles to encode (CSV or TXT)")
    parser.add_argument("--locus", help="Generate embeddings for specific locus only")
    parser.add_argument("--model", help="ProtBERT model name or path")
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
        os.path.join(data_dir, "embeddings")
    )
    model_name = args.model or config.get("model.protbert_model", "Rostlab/prot_bert")
    device = args.device or config.get("encoder.default_device", "auto")
    batch_size = args.batch_size or config.get("model.batch_size", 8)
    
    # Initialize encoder
    logger.info(f"Initializing ProtBERT encoder (model={model_name}, device={device})")
    encoder = ProtBERTEncoder(
        sequence_file=sequences_file,
        cache_dir=cache_dir,
        model_name=model_name,
        locus=args.locus,
        device=device,
        pooling_strategy=config.get("model.pooling_strategy", "mean"),
        use_peptide_binding_region=config.get("model.use_peptide_binding_region", True)
    )
    
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
        return
    
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
        return
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(alleles)} alleles in batches of {batch_size}...")
    
    # Use batch encoding for better performance
    logger.info("Performing batch encoding...")
    allele_sequences = []
    
    # Collect sequences for alleles
    for allele in alleles:
        try:
            sequence = encoder.get_sequence(allele)
            if sequence:
                allele_sequences.append((allele, sequence))
            else:
                logger.warning(f"No sequence found for {allele}")
        except Exception as e:
            logger.error(f"Error getting sequence for {allele}: {e}")
    
    # Perform batch encoding
    if allele_sequences:
        # Extract sequences
        batch_alleles = [a for a, _ in allele_sequences]
        batch_sequences = [s for _, s in allele_sequences]
        
        # Encode in batches
        for i in range(0, len(batch_sequences), batch_size):
            batch = batch_sequences[i:i + batch_size]
            batch_names = batch_alleles[i:i + batch_size]
            
            logger.info(f"Encoding batch {i//batch_size + 1}/{(len(batch_sequences)-1)//batch_size + 1} ({len(batch)} alleles)")
            
            try:
                embeddings = encoder.batch_encode(batch, batch_size=batch_size)
                
                # Cache embeddings
                for j, allele in enumerate(batch_names):
                    encoder.embeddings[allele] = embeddings[j]
                
                # Save cache periodically
                if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= len(batch_sequences):
                    logger.info(f"Saving embeddings cache ({len(encoder.embeddings)} total embeddings)")
                    encoder._save_embedding_cache()
                    
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
    
    # Save final cache
    logger.info(f"Saving final embeddings cache ({len(encoder.embeddings)} total embeddings)")
    encoder._save_embedding_cache()
    
    logger.info("Embedding generation complete")

if __name__ == "__main__":
    main()
