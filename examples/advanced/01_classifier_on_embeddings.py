#!/usr/bin/env python
"""
Advanced Example: Building a Classifier on HLA Embeddings
==========================================================

This example demonstrates how to:
1. Generate embeddings for a set of HLA alleles
2. Create labels based on biological properties
3. Train a classifier to predict properties from embeddings
4. Evaluate the classifier and visualize results

Use Case: Predicting HLA allele properties (e.g., serotype, supertype)
from protein embeddings alone.

Example Usage:
    python examples/advanced/01_classifier_on_embeddings.py
    python examples/advanced/01_classifier_on_embeddings.py --encoder esm
"""

import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.encoders import ProtBERTEncoder, ESMEncoder


def extract_serotype(allele: str) -> str:
    """
    Extract serotype from allele name.
    E.g., A*01:01 -> A1, A*02:01 -> A2
    """
    parts = allele.split('*')
    if len(parts) != 2:
        return "Unknown"
    
    locus = parts[0]
    fields = parts[1].split(':')
    if len(fields) < 1:
        return "Unknown"
    
    # Get first field (serotype)
    serotype = fields[0]
    return f"{locus}{serotype}"


def get_labeled_data(encoder, locus='A', max_per_serotype=20):
    """
    Generate embeddings and labels for HLA alleles.
    
    Args:
        encoder: HLA encoder instance
        locus: Which locus to use (A, B, C, etc.)
        max_per_serotype: Maximum alleles per serotype for balanced dataset
    
    Returns:
        embeddings: numpy array of embeddings
        labels: list of serotype labels
        alleles: list of allele names
    """
    print(f"Generating labeled dataset for locus {locus}...")
    
    # Get all alleles for this locus from the encoder's sequences
    all_alleles = [
        allele for allele in encoder.sequences.keys() 
        if allele.startswith(f"{locus}*")
    ]
    
    print(f"Found {len(all_alleles)} alleles for locus {locus}")
    
    # Group by serotype
    serotype_groups = {}
    for allele in all_alleles:
        serotype = extract_serotype(allele)
        if serotype not in serotype_groups:
            serotype_groups[serotype] = []
        serotype_groups[serotype].append(allele)
    
    # Balance dataset
    selected_alleles = []
    for serotype, alleles in serotype_groups.items():
        # Take up to max_per_serotype from each group
        selected_alleles.extend(alleles[:max_per_serotype])
    
    print(f"Selected {len(selected_alleles)} alleles across {len(serotype_groups)} serotypes")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_dict = encoder.batch_encode_alleles(selected_alleles)
    
    # Convert to arrays
    embeddings = []
    labels = []
    valid_alleles = []
    
    for allele in selected_alleles:
        if allele in embeddings_dict and embeddings_dict[allele] is not None:
            embeddings.append(embeddings_dict[allele])
            labels.append(extract_serotype(allele))
            valid_alleles.append(allele)
    
    embeddings = np.array(embeddings)
    
    print(f"Generated {len(embeddings)} valid embeddings")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return embeddings, labels, valid_alleles


def train_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest classifier."""
    print("\nTraining Random Forest classifier...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf, y_pred


def visualize_results(X_train, y_train, X_test, y_test, y_pred, output_dir):
    """Visualize the classification results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Predicted vs Actual Serotypes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    plt.close()
    
    # 2. PCA visualization
    print("\nGenerating PCA visualization...")
    pca = PCA(n_components=2)
    
    # Fit on training data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training data
    unique_labels = np.unique(y_train)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(y_train) == label
        ax1.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                   c=[color], label=label, alpha=0.6, s=50)
    
    ax1.set_title('Training Data (PCA)')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Test data with predictions
    for label, color in zip(unique_labels, colors):
        mask = np.array(y_test) == label
        ax2.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=[color], label=label, alpha=0.6, s=50)
        
    # Mark misclassifications
    misclassified = np.array(y_test) != np.array(y_pred)
    if misclassified.any():
        ax2.scatter(X_test_pca[misclassified, 0], X_test_pca[misclassified, 1],
                   c='red', marker='x', s=200, linewidths=2, label='Misclassified')
    
    ax2.set_title('Test Data (PCA) - X marks misclassifications')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_visualization.png', dpi=300, bbox_inches='tight')
    print(f"Saved PCA visualization to {output_dir / 'pca_visualization.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train classifier on HLA embeddings')
    parser.add_argument('--sequence-file', type=Path,
                       default=Path('./data/processed/hla_sequences.pkl'),
                       help='Path to HLA sequences')
    parser.add_argument('--encoder', type=str, choices=['protbert', 'esm'],
                       default='protbert', help='Which encoder to use')
    parser.add_argument('--locus', type=str, default='A',
                       help='Which locus to classify (A, B, C, etc.)')
    parser.add_argument('--max-per-serotype', type=int, default=20,
                       help='Maximum alleles per serotype')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Fraction of data for testing')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('./examples/advanced/classifier_results'),
                       help='Output directory for results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check sequence file
    if not args.sequence_file.exists():
        print(f"ERROR: Sequence file not found: {args.sequence_file}")
        print("Please run: python scripts/update_imgt.py")
        return 1
    
    # Initialize encoder
    print(f"Initializing {args.encoder} encoder...")
    if args.encoder == 'protbert':
        encoder = ProtBERTEncoder(sequence_file=str(args.sequence_file))
    else:
        encoder = ESMEncoder(sequence_file=str(args.sequence_file))
    
    # Generate labeled dataset
    embeddings, labels, alleles = get_labeled_data(
        encoder, 
        locus=args.locus,
        max_per_serotype=args.max_per_serotype
    )
    
    # Split data
    X_train, X_test, y_train, y_test, alleles_train, alleles_test = train_test_split(
        embeddings, labels, alleles,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train classifier
    clf, y_pred = train_classifier(X_train, y_train, X_test, y_test)
    
    # Visualize results
    visualize_results(X_train, y_train, X_test, y_test, y_pred, args.output_dir)
    
    # Save model and results
    results = {
        'encoder': args.encoder,
        'locus': args.locus,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_alleles': alleles_test,
        'predictions': y_pred,
        'true_labels': y_test,
        'classifier': clf
    }
    
    with open(args.output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {args.output_dir}")
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
