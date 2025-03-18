#!/usr/bin/env python
"""
Train Clinical Predictor
----------------------
Script to train a clinical prediction model using HLA embeddings.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.models.protbert import ProtBERTEncoder
from src.models.predictors import (
    HLAPredictor, 
    TransplantOutcomePredictor,
    GVHDRiskPredictor, 
    EngrafdmentPredictor
)
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def load_clinical_data(
    file_path: Path,
    donor_cols: List[str] = None,
    recipient_cols: List[str] = None,
    outcome_col: str = None
) -> Tuple[List[Tuple[List[str], List[str], Dict]], List[float]]:
    """Load clinical data from file
    
    Args:
        file_path: Path to clinical data file
        donor_cols: Column names for donor HLA alleles
        recipient_cols: Column names for recipient HLA alleles
        outcome_col: Column name for outcome variable
        
    Returns:
        Tuple of (samples, labels) where:
        - samples: List of (donor_alleles, recipient_alleles, clinical_data) tuples
        - labels: List of outcome values
    """
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return [], []
    
    # Default column names if not specified
    if donor_cols is None:
        donor_cols = ["donor_a1", "donor_a2", "donor_b1", "donor_b2", "donor_c1", "donor_c2", 
                     "donor_drb1_1", "donor_drb1_2", "donor_dqb1_1", "donor_dqb1_2"]
    
    if recipient_cols is None:
        recipient_cols = ["recipient_a1", "recipient_a2", "recipient_b1", "recipient_b2", "recipient_c1", "recipient_c2",
                         "recipient_drb1_1", "recipient_drb1_2", "recipient_dqb1_1", "recipient_dqb1_2"]
    
    if outcome_col is None:
        # Try to find outcome column by common names
        potential_outcomes = ["outcome", "survival", "gvhd", "engraftment", "rejection", "relapse", "target"]
    
    # Load data
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(file_path)
    elif suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif suffix == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        logger.error(f"Unsupported file format: {suffix}")
        return [], []
    
    # Find outcome column if not specified
    if outcome_col is None:
        for col in potential_outcomes:
            if col in df.columns:
                outcome_col = col
                logger.info(f"Using '{outcome_col}' as outcome column")
                break
        
        if outcome_col is None:
            logger.error("No outcome column specified or found")
            return [], []
    
    # Prepare data samples
    samples = []
    labels = []
    
    # Validate columns exist
    existing_donor_cols = [col for col in donor_cols if col in df.columns]
    existing_recipient_cols = [col for col in recipient_cols if col in df.columns]
    
    if not existing_donor_cols:
        logger.error(f"No donor columns found in data. Available columns: {df.columns.tolist()}")
        return [], []
        
    if not existing_recipient_cols:
        logger.error(f"No recipient columns found in data. Available columns: {df.columns.tolist()}")
        return [], []
    
    if outcome_col not in df.columns:
        logger.error(f"Outcome column '{outcome_col}' not found in data")
        return [], []
    
    # Process each row
    for _, row in df.iterrows():
        # Extract donor alleles
        donor_alleles = []
        for col in existing_donor_cols:
            if not pd.isna(row[col]) and row[col]:
                donor_alleles.append(str(row[col]).strip())
        
        # Extract recipient alleles
        recipient_alleles = []
        for col in existing_recipient_cols:
            if not pd.isna(row[col]) and row[col]:
                recipient_alleles.append(str(row[col]).strip())
        
        # Skip if no alleles found
        if not donor_alleles or not recipient_alleles:
            continue
        
        # Extract outcome
        if pd.isna(row[outcome_col]):
            continue
            
        outcome = row[outcome_col]
        
        # Extract clinical data
        clinical_data = {}
        for col in df.columns:
            # Skip HLA columns and outcome
            if col in existing_donor_cols or col in existing_recipient_cols or col == outcome_col:
                continue
                
            if not pd.isna(row[col]):
                # Try to convert to numeric if possible
                try:
                    if isinstance(row[col], (int, float)):
                        clinical_data[col] = row[col]
                    else:
                        val = row[col]
                        if val.strip().lower() in ['yes', 'y', 'true', 't', '1']:
                            clinical_data[col] = 1
                        elif val.strip().lower() in ['no', 'n', 'false', 'f', '0']:
                            clinical_data[col] = 0
                        else:
                            clinical_data[col] = val
                except:
                    clinical_data[col] = row[col]
        
        # Add to samples
        samples.append((donor_alleles, recipient_alleles, clinical_data))
        labels.append(float(outcome))
    
    logger.info(f"Loaded {len(samples)} samples with {len(existing_donor_cols)} donor and {len(existing_recipient_cols)} recipient columns")
    return samples, labels

def main():
    """Main function to train clinical predictor"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a clinical prediction model using HLA embeddings")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--embeddings", help="Path to HLA embeddings cache")
    parser.add_argument("--training-data", dest="training_data", required=True, help="Path to training data file")
    parser.add_argument("--validation-data", dest="validation_data", help="Path to validation data file")
    parser.add_argument("--model-type", dest="model_type", choices=["mlp", "lstm", "cnn"], default="mlp", help="Model architecture")
    parser.add_argument("--predictor-type", dest="predictor_type", choices=["general", "transplant", "gvhd", "engraftment"], default="general", help="Predictor type")
    parser.add_argument("--output-model", dest="output_model", help="Path to save trained model")
    parser.add_argument("--clinical-vars", dest="clinical_vars", help="Comma-separated list of clinical variables to use")
    parser.add_argument("--donor-cols", dest="donor_cols", help="Comma-separated list of donor HLA column names")
    parser.add_argument("--recipient-cols", dest="recipient_cols", help="Comma-separated list of recipient HLA column names")
    parser.add_argument("--outcome-col", dest="outcome_col", help="Column name for outcome variable")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run model on")
    parser.add_argument("--eval-metrics", dest="eval_metrics", action="store_true", help="Calculate evaluation metrics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Determine paths and parameters
    data_dir = args.data_dir or config.get("data.base_dir", str(project_dir / "data"))
    embeddings_dir = config.get("data.embeddings_dir", os.path.join(data_dir, "embeddings"))
    sequences_file = config.get("data.sequences_file", os.path.join(data_dir, "processed", "hla_sequences.pkl"))
    model_dir = os.path.join(data_dir, "models")
    
    # Parse column names
    donor_cols = args.donor_cols.split(",") if args.donor_cols else None
    recipient_cols = args.recipient_cols.split(",") if args.recipient_cols else None
    clinical_vars = args.clinical_vars.split(",") if args.clinical_vars else None
    
    # Initialize encoder
    logger.info("Initializing HLA encoder...")
    encoder = ProtBERTEncoder(
        sequence_file=sequences_file,
        cache_dir=embeddings_dir,
        device=args.device or config.get("encoder.default_device", "auto")
    )
    
    # Load training data
    logger.info(f"Loading training data from {args.training_data}")
    train_samples, train_labels = load_clinical_data(
        Path(args.training_data),
        donor_cols=donor_cols,
        recipient_cols=recipient_cols,
        outcome_col=args.outcome_col
    )
    
    if not train_samples:
        logger.error("No training samples loaded")
        return
    
    # Load validation data if provided
    validation_samples = None
    validation_labels = None
    
    if args.validation_data:
        logger.info(f"Loading validation data from {args.validation_data}")
        validation_samples, validation_labels = load_clinical_data(
            Path(args.validation_data),
            donor_cols=donor_cols,
            recipient_cols=recipient_cols,
            outcome_col=args.outcome_col
        )
    
    # Initialize predictor based on type
    logger.info(f"Initializing {args.predictor_type} predictor with {args.model_type} architecture")
    
    if args.predictor_type == "transplant":
        predictor = TransplantOutcomePredictor(
            encoder=encoder,
            clinical_variables=clinical_vars,
            model_type=args.model_type,
            device=args.device
        )
    elif args.predictor_type == "gvhd":
        predictor = GVHDRiskPredictor(
            encoder=encoder,
            clinical_variables=clinical_vars,
            model_type=args.model_type,
            device=args.device
        )
    elif args.predictor_type == "engraftment":
        predictor = EngrafdmentPredictor(
            encoder=encoder,
            clinical_variables=clinical_vars,
            model_type=args.model_type,
            device=args.device
        )
    else:
        predictor = HLAPredictor(
            encoder=encoder,
            clinical_variables=clinical_vars,
            model_type=args.model_type,
            device=args.device
        )
    
    # Train the model
    logger.info(f"Training model with {len(train_samples)} samples...")
    history = predictor.train(
        train_data=train_samples,
        train_labels=train_labels,
        validation_data=validation_samples,
        validation_labels=validation_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save the model
    output_model_path = args.output_model or os.path.join(
        model_dir, 
        f"{args.predictor_type}_{args.model_type}_model.pt"
    )
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    logger.info(f"Saving model to {output_model_path}")
    predictor.save(output_model_path)
    
    # Calculate evaluation metrics if requested
    if args.eval_metrics:
        try:
            from src.analysis.metrics import PredictionEvaluator
            
            evaluator = PredictionEvaluator()
            
            logger.info("Evaluating model on training data...")
            train_predictions = []
            
            for sample in train_samples:
                if args.predictor_type in ["transplant", "gvhd", "engraftment"]:
                    donor_alleles, recipient_alleles, clinical_data = sample
                    pred = predictor.predict_outcome(donor_alleles, recipient_alleles, clinical_data)
                else:
                    # General predictor
                    donor_alleles, _, clinical_data = sample
                    pred = predictor.predict(donor_alleles, clinical_data)
                    
                train_predictions.append(pred)
            
            # Determine if classification or regression
            unique_labels = set(train_labels)
            is_binary = len(unique_labels) <= 2 and all(label in [0, 1] for label in unique_labels)
            
            if is_binary:
                metrics = evaluator.evaluate_classification(train_labels, train_predictions)
                logger.info("Classification metrics:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  AUC: {metrics['auc']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            else:
                metrics = evaluator.evaluate_regression(train_labels, train_predictions)
                logger.info("Regression metrics:")
                logger.info(f"  MSE: {metrics['mse']:.4f}")
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  R²: {metrics['r2']:.4f}")
                
            # Evaluate on validation data if available
            if validation_samples and validation_labels:
                logger.info("Evaluating model on validation data...")
                val_predictions = []
                
                for sample in validation_samples:
                    if args.predictor_type in ["transplant", "gvhd", "engraftment"]:
                        donor_alleles, recipient_alleles, clinical_data = sample
                        pred = predictor.predict_outcome(donor_alleles, recipient_alleles, clinical_data)
                    else:
                        # General predictor
                        donor_alleles, _, clinical_data = sample
                        pred = predictor.predict(donor_alleles, clinical_data)
                        
                    val_predictions.append(pred)
                
                if is_binary:
                    metrics = evaluator.evaluate_classification(validation_labels, val_predictions)
                    logger.info("Validation classification metrics:")
                    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  AUC: {metrics['auc']:.4f}")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
                else:
                    metrics = evaluator.evaluate_regression(validation_labels, val_predictions)
                    logger.info("Validation regression metrics:")
                    logger.info(f"  MSE: {metrics['mse']:.4f}")
                    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"  MAE: {metrics['mae']:.4f}")
                    logger.info(f"  R²: {metrics['r2']:.4f}")
                    
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {e}")
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()
