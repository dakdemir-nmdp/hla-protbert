#!/usr/bin/env python
"""
Multi-Encoder Comparison Example
--------------------------------
Demonstrates how to use all available encoders (ProtBERT, ESM-2, ProtT5, Ankh)
and compare their embeddings for HLA alleles.

This example shows:
1. How to initialize each encoder
2. Generate embeddings with different models
3. Compare embedding similarities
4. Create ensemble embeddings
5. Performance benchmarking
"""
import os
import sys
import argparse
import logging
import numpy as np
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder
from hlaprotbert.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def initialize_encoders(
    sequence_file: str,
    data_dir: str,
    device: str = None
) -> Dict[str, object]:
    """Initialize all available encoders.
    
    Args:
        sequence_file: Path to HLA sequences pickle file
        data_dir: Base data directory for caching
        device: Device to use ('cpu', 'cuda', or None for auto)
        
    Returns:
        Dictionary mapping encoder names to encoder instances
    """
    encoders = {}
    
    logger.info("Initializing encoders...")
    
    # ProtBERT
    try:
        logger.info("  Loading ProtBERT (420M params)...")
        encoders['protbert'] = ProtBERTEncoder(
            sequence_file=sequence_file,
            cache_dir=f"{data_dir}/embeddings/protbert",
            device=device
        )
        logger.info("    ✓ ProtBERT ready")
    except Exception as e:
        logger.warning(f"    ✗ Failed to load ProtBERT: {e}")
    
    # ESM-2
    try:
        logger.info("  Loading ESM-2 (650M params)...")
        encoders['esm'] = ESMEncoder(
            sequence_file=sequence_file,
            cache_dir=f"{data_dir}/embeddings/esm",
            device=device
        )
        logger.info("    ✓ ESM-2 ready")
    except Exception as e:
        logger.warning(f"    ✗ Failed to load ESM-2: {e}")
    
    # ProtT5
    try:
        logger.info("  Loading ProtT5 (1.3B params)...")
        encoders['prott5'] = ProtT5Encoder(
            sequence_file=sequence_file,
            cache_dir=f"{data_dir}/embeddings/prott5",
            device=device
        )
        logger.info("    ✓ ProtT5 ready")
    except Exception as e:
        logger.warning(f"    ✗ Failed to load ProtT5: {e}")
    
    # Ankh Base
    try:
        logger.info("  Loading Ankh Base (50M params)...")
        encoders['ankh_base'] = AnkhEncoder(
            sequence_file=sequence_file,
            cache_dir=f"{data_dir}/embeddings/ankh",
            model_variant="base",
            device=device
        )
        logger.info("    ✓ Ankh Base ready")
    except Exception as e:
        logger.warning(f"    ✗ Failed to load Ankh Base: {e}")
    
    # Ankh Large
    try:
        logger.info("  Loading Ankh Large (650M params)...")
        encoders['ankh_large'] = AnkhEncoder(
            sequence_file=sequence_file,
            cache_dir=f"{data_dir}/embeddings/ankh",
            model_variant="large",
            device=device
        )
        logger.info("    ✓ Ankh Large ready")
    except Exception as e:
        logger.warning(f"    ✗ Failed to load Ankh Large: {e}")
    
    logger.info(f"Successfully initialized {len(encoders)}/{5} encoders")
    return encoders


def encode_with_timing(encoder: object, allele: str) -> tuple:
    """Encode an allele and measure time.
    
    Args:
        encoder: Encoder instance
        allele: HLA allele identifier
        
    Returns:
        Tuple of (embedding, elapsed_time)
    """
    start_time = time.time()
    embedding = encoder.get_embedding(allele)
    elapsed_time = time.time() - start_time
    return embedding, elapsed_time


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize vectors
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    return float(similarity)


def create_ensemble_embedding(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Create ensemble embedding by concatenating normalized embeddings.
    
    Args:
        embeddings: Dictionary mapping encoder names to embeddings
        
    Returns:
        Concatenated ensemble embedding
    """
    from sklearn.preprocessing import normalize
    
    # Normalize each embedding
    normalized = []
    for name, emb in embeddings.items():
        emb_norm = normalize(emb.reshape(1, -1))[0]
        normalized.append(emb_norm)
    
    # Concatenate
    ensemble = np.concatenate(normalized)
    return ensemble


def main():
    """Main function demonstrating multi-encoder comparison"""
    parser = argparse.ArgumentParser(
        description="Compare multiple protein language models for HLA encoding"
    )
    parser.add_argument(
        "--alleles",
        required=True,
        nargs='+',
        help="HLA alleles to encode (space separated)"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Base data directory"
    )
    parser.add_argument(
        "--sequence-file",
        default="./data/processed/hla_sequences.pkl",
        help="Path to HLA sequences pickle file"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help="Device to use for encoding"
    )
    parser.add_argument(
        "--compare-alleles",
        action="store_true",
        help="Compare similarity between provided alleles"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Create ensemble embeddings"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Check if sequence file exists
    if not Path(args.sequence_file).exists():
        logger.error(f"Sequence file not found: {args.sequence_file}")
        logger.info("Please run: python scripts/update_imgt.py")
        return 1
    
    # Initialize encoders
    device = None if args.device == 'auto' else args.device
    encoders = initialize_encoders(args.sequence_file, args.data_dir, device)
    
    if not encoders:
        logger.error("No encoders were successfully initialized!")
        return 1
    
    # Encode each allele with each encoder
    print("\n" + "="*80)
    print("ENCODING ALLELES")
    print("="*80)
    
    all_embeddings = {allele: {} for allele in args.alleles}
    
    for allele in args.alleles:
        print(f"\nEncoding: {allele}")
        print("-" * 80)
        
        for name, encoder in encoders.items():
            try:
                embedding, elapsed = encode_with_timing(encoder, allele)
                all_embeddings[allele][name] = embedding
                
                print(f"  {name:12s}: shape={embedding.shape} time={elapsed:.3f}s")
            except Exception as e:
                logger.error(f"  {name:12s}: Failed - {e}")
    
    # Compare alleles if requested
    if args.compare_alleles and len(args.alleles) >= 2:
        print("\n" + "="*80)
        print("INTER-ALLELE SIMILARITY")
        print("="*80)
        
        for i, allele1 in enumerate(args.alleles):
            for allele2 in args.alleles[i+1:]:
                print(f"\nComparing {allele1} vs {allele2}:")
                print("-" * 80)
                
                for name in encoders.keys():
                    if name in all_embeddings[allele1] and name in all_embeddings[allele2]:
                        emb1 = all_embeddings[allele1][name]
                        emb2 = all_embeddings[allele2][name]
                        similarity = compute_similarity(emb1, emb2)
                        print(f"  {name:12s}: {similarity:.4f}")
    
    # Create ensemble embeddings if requested
    if args.ensemble and len(args.alleles) >= 1:
        print("\n" + "="*80)
        print("ENSEMBLE EMBEDDINGS")
        print("="*80)
        
        for allele in args.alleles:
            if len(all_embeddings[allele]) >= 2:
                ensemble = create_ensemble_embedding(all_embeddings[allele])
                print(f"\n{allele}:")
                print(f"  Individual embeddings: {len(all_embeddings[allele])}")
                print(f"  Ensemble shape: {ensemble.shape}")
                print(f"  Constituent dimensions:")
                for name, emb in all_embeddings[allele].items():
                    print(f"    {name:12s}: {emb.shape[0]}")
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK")
        print("="*80)
        
        benchmark_allele = args.alleles[0]
        n_iterations = 5
        
        print(f"\nBenchmarking with allele: {benchmark_allele}")
        print(f"Iterations: {n_iterations}")
        print("-" * 80)
        
        for name, encoder in encoders.items():
            times = []
            for _ in range(n_iterations):
                _, elapsed = encode_with_timing(encoder, benchmark_allele)
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"  {name:12s}: {avg_time:.3f}s ± {std_time:.3f}s")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSuccessfully encoded {len(args.alleles)} allele(s)")
    print(f"Using {len(encoders)} encoder(s): {', '.join(encoders.keys())}")
    print("\nModel Specifications:")
    print(f"  ProtBERT:    420M params, 768-dim embeddings")
    print(f"  ESM-2:       650M params, 1280-dim embeddings")
    print(f"  ProtT5:      1.3B params, 1024-dim embeddings")
    print(f"  Ankh Base:   50M params, 768-dim embeddings")
    print(f"  Ankh Large:  650M params, 1536-dim embeddings")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
