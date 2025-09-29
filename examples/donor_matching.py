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

from src.models.encoders import ProtBERTEncoder, ESMEncoder # Import both
from src.models.encoder import HLAEncoder # Import base class for type hinting
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

    # Add encoder selection arguments
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["protbert", "esm"],
        default="protbert",
        help="Type of encoder model to use ('protbert' or 'esm')"
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
        help="Pooling strategy for embeddings ('mean' or 'cls') - primarily for ESM"
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
    
    # Standardize loci names in HLA alleles
    donor_alleles = args.donor
    recipient_alleles = args.recipient
    
    # Initialize encoder based on type
    logger.info(f"Initializing {args.encoder_type.upper()} encoder...")
    try:
        encoder_args = {
            "sequence_file": sequences_file,
            "cache_dir": embeddings_dir,
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
        elif args.encoder_type == "esm":
            EncoderClass = ESMEncoder
            # ESM specific args
            encoder_args["pooling_strategy"] = args.pooling_strategy
            # Add HF token from config if available
            hf_token = config.get("model.hf_token", None)
            if hf_token:
                encoder_args["hf_token"] = hf_token
            if not args.model_name:
                 encoder_args["model_name"] = config.get("model.esm_model_name", "facebook/esm2_t33_650M_UR50D")
        else:
            # This case should not be reached due to argparse choices
            raise ValueError(f"Unsupported encoder type: {args.encoder_type}")

        encoder: HLAEncoder = EncoderClass(**encoder_args) # Add type hint

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
    print(f"Using {args.encoder_type.upper()} model: {encoder.model_name}")
    print(f"Pooling strategy: {encoder.pooling_strategy}")
    if args.encoder_type == "protbert":
        print(f"Using peptide binding region: {encoder.use_peptide_binding_region}")
    print(f"Device: {encoder.device}")
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
