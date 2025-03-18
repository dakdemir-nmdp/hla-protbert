#!/usr/bin/env python
"""
HLA Matching Example
------------------
Demonstrates how to use the HLA-ProtBERT system for donor-recipient matching analysis.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.models.protbert import ProtBERTEncoder
from src.analysis.matching import MatchingAnalyzer
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def main():
    """Main function demonstrating donor-recipient matching analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze HLA matching between donor and recipient")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--donor", required=True, nargs='+', help="Donor HLA alleles (space separated)")
    parser.add_argument("--recipient", required=True, nargs='+', help="Recipient HLA alleles (space separated)")
    parser.add_argument("--report", help="Save matching report to file (PDF)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for functional matching")
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
    
    # Standardize loci names in HLA alleles
    donor_alleles = args.donor
    recipient_alleles = args.recipient
    
    # Initialize encoder
    logger.info("Initializing HLA encoder...")
    try:
        encoder = ProtBERTEncoder(
            sequence_file=sequences_file,
            cache_dir=embeddings_dir
        )
    except FileNotFoundError:
        logger.error(f"Sequence file not found: {sequences_file}")
        logger.error("Please run scripts/update_imgt.py and scripts/generate_embeddings.py first.")
        return
    
    # Initialize matching analyzer
    logger.info("Performing matching analysis...")
    analyzer = MatchingAnalyzer(
        encoder=encoder,
        similarity_threshold=args.threshold
    )
    
    # Analyze matching
    results = analyzer.analyze_matching(donor_alleles, recipient_alleles)
    
    # Print matching summary
    print("\n===== HLA Matching Analysis =====\n")
    print(f"Donor HLA: {', '.join(donor_alleles)}")
    print(f"Recipient HLA: {', '.join(recipient_alleles)}")
    print(f"\nCommon Loci: {', '.join(results['common_loci'])}")
    print(f"Exact Matches: {len(results['exact_matches'])} ({100*results['exact_match_pct']:.1f}%)")
    print(f"Functional Matches: {len(results['functional_matches'])} (additional {100*(results['functional_match_pct']-results['exact_match_pct']):.1f}%)")
    print(f"Total Match: {100*results['functional_match_pct']:.1f}%")
    print(f"Average Similarity: {results['average_similarity']:.3f}")
    
    # Print exact matches
    if results['exact_matches']:
        print("\nExact Matches:")
        for locus, allele in results['exact_matches']:
            print(f"  {allele}")
    
    # Print functional matches
    if results['functional_matches']:
        print("\nFunctional Matches (high similarity but not exact):")
        for locus, r_allele, d_allele, similarity in results['functional_matches']:
            print(f"  {r_allele} ~ {d_allele} (similarity: {similarity:.3f})")
    
    # Print mismatches
    if results['mismatches']:
        print("\nMismatches:")
        for r_allele, d_allele, similarity in results['mismatches']:
            print(f"  {r_allele} vs {d_allele or 'none'} (similarity: {similarity:.3f})")
    
    # Generate report if requested
    if args.report:
        logger.info(f"Generating matching report: {args.report}")
        analyzer.generate_report(
            donor_alleles=donor_alleles,
            recipient_alleles=recipient_alleles,
            output_file=args.report
        )
        print(f"\nDetailed report saved to: {args.report}")
    
    # Generate visualizations if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            logger.info("Generating matching visualizations...")
            
            # Create visualization
            fig = analyzer.visualize_matching(results)
            
            # Show visualization
            plt.show()
            
        except ImportError:
            logger.error("Matplotlib not installed; cannot generate visualizations")
            print("\nCannot generate visualizations. Please install matplotlib:\n  pip install matplotlib")

if __name__ == "__main__":
    main()
