#!/usr/bin/env python
"""
Advanced Example: Comparing Pooling Strategies
===============================================

This example demonstrates how different pooling strategies affect
the quality and characteristics of HLA allele embeddings.

Pooling strategies convert variable-length token embeddings into
fixed-size representations. Common strategies include:
- Mean pooling: Average across all tokens
- Max pooling: Maximum value per dimension
- CLS token: Use the [CLS] token embedding (BERT models)
- Last token: Use the final token

This example:
1. Generates embeddings using different pooling strategies
2. Compares how well they preserve biological relationships
3. Visualizes the differences
4. Provides recommendations

Usage:
    python examples/advanced/03_pooling_comparison.py
    python examples/advanced/03_pooling_comparison.py --alleles A*01:01 A*02:01 B*07:02
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.encoders import ProtBERTEncoder


def get_embeddings_with_pooling(encoder_class, sequence_file, alleles, pooling_strategy):
    """Get embeddings using specified pooling strategy."""
    encoder = encoder_class(
        sequence_file=str(sequence_file),
        pooling_strategy=pooling_strategy
    )
    
    embeddings = {}
    for allele in alleles:
        emb = encoder.get_embedding(allele, force=True)
        if emb is not None:
            embeddings[allele] = emb
    
    return embeddings


def compare_similarity_preservation(alleles, embeddings_by_pooling):
    """
    Compare how well each pooling strategy preserves biological relationships.
    
    Returns similarity matrices for each pooling strategy.
    """
    similarity_matrices = {}
    
    for pooling, embeddings in embeddings_by_pooling.items():
        n = len(alleles)
        sim_matrix = np.zeros((n, n))
        
        for i, allele1 in enumerate(alleles):
            for j, allele2 in enumerate(alleles):
                if allele1 in embeddings and allele2 in embeddings:
                    emb1 = embeddings[allele1]
                    emb2 = embeddings[allele2]
                    # Cosine similarity (1 - cosine distance)
                    sim_matrix[i, j] = 1 - cosine(emb1, emb2)
        
        similarity_matrices[pooling] = sim_matrix
    
    return similarity_matrices


def calculate_embedding_statistics(embeddings):
    """Calculate statistics about embedding distributions."""
    all_embs = np.array(list(embeddings.values()))
    
    stats = {
        'mean': np.mean(all_embs, axis=0).mean(),
        'std': np.std(all_embs, axis=0).mean(),
        'min': all_embs.min(),
        'max': all_embs.max(),
        'l2_norm_mean': np.linalg.norm(all_embs, axis=1).mean(),
        'l2_norm_std': np.linalg.norm(all_embs, axis=1).std(),
    }
    
    return stats


def visualize_pooling_comparison(alleles, embeddings_by_pooling, similarity_matrices, output_dir):
    """Create visualizations comparing pooling strategies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pooling_strategies = list(embeddings_by_pooling.keys())
    n_strategies = len(pooling_strategies)
    
    # 1. Similarity Matrices Heatmap
    fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
    if n_strategies == 1:
        axes = [axes]
    
    for ax, pooling in zip(axes, pooling_strategies):
        sim_matrix = similarity_matrices[pooling]
        sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=alleles, yticklabels=alleles, ax=ax,
                   vmin=0, vmax=1, square=True)
        ax.set_title(f'{pooling.upper()} Pooling\nSimilarity Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Saved similarity matrices to {output_dir / 'similarity_matrices.png'}")
    plt.close()
    
    # 2. Embedding Statistics Comparison
    stats_data = []
    for pooling, embeddings in embeddings_by_pooling.items():
        stats = calculate_embedding_statistics(embeddings)
        stats['pooling'] = pooling
        stats_data.append(stats)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['mean', 'std', 'min', 'max', 'l2_norm_mean', 'l2_norm_std']
    for ax, metric in zip(axes, metrics):
        values = [s[metric] for s in stats_data]
        labels = [s['pooling'] for s in stats_data]
        
        ax.bar(labels, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(labels))))
        ax.set_title(f'Embedding {metric.replace("_", " ").title()}')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Saved embedding statistics to {output_dir / 'embedding_statistics.png'}")
    plt.close()
    
    # 3. t-SNE Visualization (if we have enough alleles)
    if len(alleles) >= 10:
        fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
        if n_strategies == 1:
            axes = [axes]
        
        for ax, pooling in zip(axes, pooling_strategies):
            embeddings = embeddings_by_pooling[pooling]
            emb_array = np.array([embeddings[a] for a in alleles if a in embeddings])
            
            if len(emb_array) >= 10:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_array)-1))
                coords = tsne.fit_transform(emb_array)
                
                scatter = ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6, 
                                   c=range(len(alleles)), cmap='tab20')
                
                # Add labels
                for i, allele in enumerate(alleles):
                    if allele in embeddings:
                        ax.annotate(allele, (coords[i, 0], coords[i, 1]),
                                  fontsize=8, alpha=0.7)
                
                ax.set_title(f'{pooling.upper()} Pooling\nt-SNE Visualization')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE comparison to {output_dir / 'tsne_comparison.png'}")
        plt.close()
    
    # 4. Hierarchical Clustering Dendrograms
    fig, axes = plt.subplots(1, n_strategies, figsize=(8*n_strategies, 6))
    if n_strategies == 1:
        axes = [axes]
    
    for ax, pooling in zip(axes, pooling_strategies):
        embeddings = embeddings_by_pooling[pooling]
        emb_array = np.array([embeddings[a] for a in alleles if a in embeddings])
        
        linkage_matrix = linkage(emb_array, method='ward')
        dendro = dendrogram(linkage_matrix, labels=alleles, ax=ax, 
                          leaf_font_size=10, leaf_rotation=90)
        ax.set_title(f'{pooling.upper()} Pooling\nHierarchical Clustering')
        ax.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_dendrograms.png', dpi=300, bbox_inches='tight')
    print(f"Saved clustering dendrograms to {output_dir / 'clustering_dendrograms.png'}")
    plt.close()


def generate_recommendations(similarity_matrices, embeddings_by_pooling):
    """Generate recommendations based on the comparison."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Calculate variance in similarity for each pooling
    variances = {}
    for pooling, sim_matrix in similarity_matrices.items():
        # Variance of off-diagonal elements (how much similarity varies)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
        variance = np.var(sim_matrix[mask])
        variances[pooling] = variance
    
    print("\nSimilarity Variance (higher = more discriminative):")
    for pooling, var in sorted(variances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pooling}: {var:.4f}")
    
    # L2 norm statistics
    print("\nEmbedding L2 Norms:")
    for pooling, embeddings in embeddings_by_pooling.items():
        stats = calculate_embedding_statistics(embeddings)
        print(f"  {pooling}: {stats['l2_norm_mean']:.2f} ± {stats['l2_norm_std']:.2f}")
    
    # Recommendations
    print("\n" + "-"*70)
    print("General Guidelines:")
    print("-"*70)
    print("• MEAN pooling: Good default, stable across sequence lengths")
    print("• MAX pooling: Emphasizes salient features, can be noisy")
    print("• CLS token: Works well for BERT models trained with [CLS]")
    print("• Higher variance = more discriminative embeddings")
    print("• Lower L2 norm variability = more stable embeddings")
    
    # Pick best based on variance
    best_pooling = max(variances.items(), key=lambda x: x[1])[0]
    print(f"\nFor this dataset, {best_pooling.upper()} pooling shows highest variance")
    print("(most discriminative between alleles)")


def main():
    parser = argparse.ArgumentParser(description='Compare pooling strategies for HLA embeddings')
    parser.add_argument('--sequence-file', type=Path,
                       default=Path('./data/processed/hla_sequences.pkl'),
                       help='Path to HLA sequences')
    parser.add_argument('--alleles', nargs='+',
                       default=['A*01:01', 'A*01:02', 'A*02:01', 'A*03:01', 
                               'B*07:02', 'B*08:01', 'C*07:01', 'C*07:02'],
                       help='Alleles to compare')
    parser.add_argument('--pooling-strategies', nargs='+',
                       default=['mean', 'max'],
                       help='Pooling strategies to compare')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('./examples/advanced/pooling_comparison_results'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Check sequence file
    if not args.sequence_file.exists():
        print(f"ERROR: Sequence file not found: {args.sequence_file}")
        print("Please run: python scripts/update_imgt.py")
        return 1
    
    print(f"Comparing pooling strategies: {args.pooling_strategies}")
    print(f"Using {len(args.alleles)} alleles: {args.alleles}\n")
    
    # Generate embeddings with each pooling strategy
    embeddings_by_pooling = {}
    
    for pooling in args.pooling_strategies:
        print(f"Generating embeddings with {pooling} pooling...")
        try:
            embeddings = get_embeddings_with_pooling(
                ProtBERTEncoder,
                args.sequence_file,
                args.alleles,
                pooling
            )
            embeddings_by_pooling[pooling] = embeddings
            print(f"  Generated {len(embeddings)} embeddings")
        except Exception as e:
            print(f"  ERROR with {pooling} pooling: {e}")
    
    if not embeddings_by_pooling:
        print("ERROR: No pooling strategies succeeded")
        return 1
    
    # Compare similarity preservation
    print("\nComparing similarity preservation...")
    similarity_matrices = compare_similarity_preservation(args.alleles, embeddings_by_pooling)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_pooling_comparison(args.alleles, embeddings_by_pooling, 
                                similarity_matrices, args.output_dir)
    
    # Generate recommendations
    generate_recommendations(similarity_matrices, embeddings_by_pooling)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
