#!/usr/bin/env python
"""
Benchmark Suite for HLA-ProtBERT Encoders
==========================================

Measures and compares performance of all supported encoders:
- ProtBERT
- ESM-2
- ProtT5
- Ankh (base and large)

Usage:
    python benchmarks/benchmark_encoders.py --output benchmarks/results.json
    python benchmarks/benchmark_encoders.py --quick  # Fast test with fewer iterations
    python benchmarks/benchmark_encoders.py --encoder protbert  # Test single encoder

Requirements:
    - HLA sequences must be downloaded (run scripts/update_imgt.py first)
    - All encoder models will be downloaded on first run (~15GB total)
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoders import ProtBERTEncoder, ESMEncoder, ProtT5Encoder, AnkhEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': torch.get_num_threads(),
        'torch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
    else:
        info['cuda_available'] = False
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['mps_available'] = True
    else:
        info['mps_available'] = False
    
    return info


def benchmark_single_encoding(encoder, allele: str, n_runs: int = 10) -> Dict[str, float]:
    """Benchmark single allele encoding."""
    times = []
    
    # Warmup
    encoder.get_embedding(allele, force=True)
    
    # Actual benchmark
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder.get_embedding(allele, force=True)
        times.append(time.perf_counter() - start)
    
    return {
        'mean_sec': float(np.mean(times)),
        'std_sec': float(np.std(times)),
        'min_sec': float(np.min(times)),
        'max_sec': float(np.max(times)),
        'median_sec': float(np.median(times))
    }


def benchmark_batch_encoding(encoder, alleles: List[str], batch_size: int, n_runs: int = 5) -> Dict[str, float]:
    """Benchmark batch encoding."""
    times = []
    
    # Warmup
    encoder.batch_encode_alleles(alleles, batch_size=batch_size, force=True)
    
    # Actual benchmark
    for _ in range(n_runs):
        start = time.perf_counter()
        encoder.batch_encode_alleles(alleles, batch_size=batch_size, force=True)
        times.append(time.perf_counter() - start)
    
    throughput = len(alleles) / np.mean(times)
    
    return {
        'mean_sec': float(np.mean(times)),
        'std_sec': float(np.std(times)),
        'min_sec': float(np.min(times)),
        'max_sec': float(np.max(times)),
        'throughput_alleles_per_sec': float(throughput)
    }


def get_test_alleles(sequence_file: Path, locus: str = "A", count: int = 50) -> List[str]:
    """Get test alleles from sequence file."""
    import pickle
    
    with open(sequence_file, 'rb') as f:
        sequences = pickle.load(f)
    
    # Get alleles for specified locus
    locus_alleles = [allele for allele in sequences.keys() if allele.startswith(f"{locus}*")]
    
    # Return subset
    return locus_alleles[:count] if len(locus_alleles) > count else locus_alleles


def benchmark_encoder(encoder_name: str, encoder_class, config: Dict, test_alleles: List[str], args) -> Dict[str, Any]:
    """Benchmark a specific encoder."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking {encoder_name}")
    logger.info(f"{'='*60}")
    
    results = {
        'encoder_name': encoder_name,
        'config': config,
        'n_test_alleles': len(test_alleles)
    }
    
    try:
        # Initialize encoder
        logger.info(f"Initializing {encoder_name}...")
        encoder = encoder_class(**config)
        
        # Get embedding dimensions
        sample_embedding = encoder.get_embedding(test_alleles[0])
        results['embedding_dim'] = len(sample_embedding)
        
        # Single allele benchmark
        logger.info("Running single allele benchmark...")
        single_results = benchmark_single_encoding(
            encoder, 
            test_alleles[0], 
            n_runs=args.n_runs
        )
        results['single_allele'] = single_results
        logger.info(f"  Mean time: {single_results['mean_sec']:.4f}s ± {single_results['std_sec']:.4f}s")
        
        # Batch benchmarks
        results['batch_encoding'] = {}
        for batch_size in args.batch_sizes:
            logger.info(f"Running batch benchmark (batch_size={batch_size})...")
            batch_results = benchmark_batch_encoding(
                encoder,
                test_alleles[:min(len(test_alleles), batch_size * 2)],
                batch_size=batch_size,
                n_runs=max(3, args.n_runs // 2)
            )
            results['batch_encoding'][f'batch_{batch_size}'] = batch_results
            logger.info(f"  Throughput: {batch_results['throughput_alleles_per_sec']:.2f} alleles/sec")
        
        results['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Error benchmarking {encoder_name}: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark HLA-ProtBERT encoders')
    parser.add_argument('--sequence-file', type=Path, 
                       default=Path('./data/processed/hla_sequences.pkl'),
                       help='Path to HLA sequences pickle file')
    parser.add_argument('--output', type=Path, 
                       default=Path('./benchmarks/results.json'),
                       help='Output file for results')
    parser.add_argument('--encoder', type=str, choices=['protbert', 'esm', 'prott5', 'ankh-base', 'ankh-large', 'all'],
                       default='all', help='Which encoder to benchmark')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for benchmarking')
    parser.add_argument('--n-runs', type=int, default=10,
                       help='Number of runs for single allele benchmark')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32],
                       help='Batch sizes to test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer iterations')
    parser.add_argument('--locus', type=str, default='A',
                       help='Locus to use for test alleles')
    parser.add_argument('--n-alleles', type=int, default=50,
                       help='Number of test alleles to use')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.n_runs = 3
        args.batch_sizes = [8]
        args.n_alleles = 10
        logger.info("Quick mode enabled - using reduced iterations")
    
    # Check sequence file exists
    if not args.sequence_file.exists():
        logger.error(f"Sequence file not found: {args.sequence_file}")
        logger.error("Please run: python scripts/update_imgt.py")
        return 1
    
    # Get test alleles
    logger.info(f"Loading test alleles from {args.sequence_file}")
    test_alleles = get_test_alleles(args.sequence_file, args.locus, args.n_alleles)
    logger.info(f"Using {len(test_alleles)} test alleles from locus {args.locus}")
    
    # Collect system info
    system_info = get_system_info()
    logger.info("\nSystem Information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    # Define encoder configurations
    base_config = {
        'sequence_file': str(args.sequence_file),
        'device': args.device if args.device != 'auto' else None
    }
    
    encoders_to_test = []
    
    if args.encoder in ['protbert', 'all']:
        encoders_to_test.append(('ProtBERT', ProtBERTEncoder, base_config))
    
    if args.encoder in ['esm', 'all']:
        encoders_to_test.append(('ESM-2', ESMEncoder, base_config))
    
    if args.encoder in ['prott5', 'all']:
        encoders_to_test.append(('ProtT5', ProtT5Encoder, base_config))
    
    if args.encoder in ['ankh-base', 'all']:
        ankh_base_config = {**base_config, 'model_variant': 'base'}
        encoders_to_test.append(('Ankh-Base', AnkhEncoder, ankh_base_config))
    
    if args.encoder in ['ankh-large', 'all']:
        ankh_large_config = {**base_config, 'model_variant': 'large'}
        encoders_to_test.append(('Ankh-Large', AnkhEncoder, ankh_large_config))
    
    # Run benchmarks
    all_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': system_info,
            'test_config': {
                'locus': args.locus,
                'n_alleles': len(test_alleles),
                'n_runs': args.n_runs,
                'batch_sizes': args.batch_sizes,
                'device': args.device
            }
        },
        'results': []
    }
    
    for encoder_name, encoder_class, config in encoders_to_test:
        result = benchmark_encoder(encoder_name, encoder_class, config, test_alleles, args)
        all_results['results'].append(result)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"{'='*60}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for result in all_results['results']:
        if result['status'] == 'success':
            single = result['single_allele']
            print(f"\n{result['encoder_name']}:")
            print(f"  Embedding Dimension: {result['embedding_dim']}")
            print(f"  Single Allele: {single['mean_sec']:.4f}s ± {single['std_sec']:.4f}s")
            for batch_name, batch_result in result['batch_encoding'].items():
                print(f"  {batch_name}: {batch_result['throughput_alleles_per_sec']:.2f} alleles/sec")
        else:
            print(f"\n{result['encoder_name']}: FAILED - {result.get('error', 'unknown error')}")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
