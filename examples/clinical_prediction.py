#!/usr/bin/env python
"""
Clinical Prediction Example
-------------------------
Demonstrates how to use the HLA-ProtBERT system for clinical outcome prediction.
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
from src.models.predictors import (
    TransplantOutcomePredictor, 
    GVHDRiskPredictor, 
    EngrafdmentPredictor
)
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def main():
    """Main function demonstrating clinical prediction using HLA embeddings"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict clinical outcomes using HLA embeddings")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--donor", required=True, nargs='+', help="Donor HLA alleles (space separated)")
    parser.add_argument("--recipient", required=True, nargs='+', help="Recipient HLA alleles (space separated)")
    parser.add_argument("--clinical", nargs='*', help="Clinical variables in format name=value")
    parser.add_argument("--prediction-type", dest="prediction_type", choices=["survival", "gvhd", "engraftment"], default="survival", help="Type of prediction to make")
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
    
    # Parse clinical variables
    clinical_data = {}
    if args.clinical:
        for item in args.clinical:
            if '=' in item:
                name, value = item.split('=', 1)
                # Try to convert to numeric if possible
                try:
                    if value.lower() in ['true', 'yes', 'y']:
                        value = 1
                    elif value.lower() in ['false', 'no', 'n']:
                        value = 0
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                except:
                    pass
                    
                clinical_data[name] = value
    
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
    
    # Choose predictor based on prediction type
    if args.prediction_type == "gvhd":
        logger.info("Loading GVHD risk predictor...")
        predictor = GVHDRiskPredictor.load(args.model, encoder)
    elif args.prediction_type == "engraftment":
        logger.info("Loading engraftment predictor...")
        predictor = EngrafdmentPredictor.load(args.model, encoder)
    else:  # default to survival
        logger.info("Loading transplant outcome predictor...")
        predictor = TransplantOutcomePredictor.load(args.model, encoder)
    
    # Make prediction
    logger.info("Making prediction...")
    
    if args.prediction_type == "engraftment":
        score = predictor.predict_engraftment(args.donor, args.recipient, clinical_data)
    else:
        score = predictor.predict_outcome(args.donor, args.recipient, clinical_data)
    
    explanation = predictor.get_outcome_explanation(score)
    
    # Print results
    print("\n===== Clinical Prediction Results =====\n")
    
    print(f"Prediction Type: {args.prediction_type.capitalize()}")
    print(f"Model: {args.model}")
    print(f"\nDonor HLA: {', '.join(args.donor)}")
    print(f"Recipient HLA: {', '.join(args.recipient)}")
    
    if clinical_data:
        print("\nClinical Variables:")
        for name, value in clinical_data.items():
            print(f"  {name}: {value}")
    
    print(f"\nPrediction Score: {score:.4f}")
    print(f"Interpretation: {explanation}")
    
    if args.prediction_type == "survival":
        print(f"\nHigher score indicates better predicted survival outcome.")
    elif args.prediction_type == "gvhd":
        print(f"\nHigher score indicates higher risk of GVHD.")
    elif args.prediction_type == "engraftment":
        print(f"\nHigher score indicates better engraftment probability.")
    
    # Provide any warnings about missing data
    missing_clinical = []
    for clinical_var in predictor.clinical_variables:
        if clinical_var not in clinical_data:
            missing_clinical.append(clinical_var)
    
    if missing_clinical:
        print("\nWarning: The following clinical variables were not provided:")
        for var in missing_clinical:
            print(f"  {var}")
        print("Predictions may be less accurate without this information.")
    
    # Show most important contributing factors
    print("\nNote: This is a demonstration only. Clinical decisions should not be based solely on this prediction.")

if __name__ == "__main__":
    main()
