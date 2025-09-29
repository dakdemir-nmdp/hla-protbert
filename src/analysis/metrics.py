"""
Evaluation Metrics
----------------
Tools for evaluating HLA embedding quality and prediction performance.
"""
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Check for optional packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed; visualization features will be limited")
    PLOTTING_AVAILABLE = False

class EmbeddingEvaluator:
    """Evaluates the quality of HLA embeddings"""
    
    def __init__(self, encoder):
        """Initialize evaluator
        
        Args:
            encoder: HLAEncoder instance
        """
        self.encoder = encoder
    
    def evaluate_consistency(self, locus: str = None) -> Dict:
        """Evaluate embedding consistency within allele groups
        
        Args:
            locus: HLA locus to evaluate (or None for all loci)
            
        Returns:
            Dict with consistency metrics
        """
        # Get embeddings for the locus (or all)
        if locus:
            embeddings = {
                allele: emb for allele, emb in self.encoder.embeddings.items()
                if allele.startswith(f"{locus}*")
            }
        else:
            embeddings = self.encoder.embeddings
            
        if not embeddings:
            logger.error(f"No embeddings found for evaluation")
            return {"error": "No embeddings found"}
            
        # Group alleles by first field
        groups = defaultdict(list)
        for allele in embeddings:
            if '*' in allele:
                parts = allele.split('*')
                if len(parts) > 1 and ':' in parts[1]:
                    group = f"{parts[0]}*{parts[1].split(':')[0]}"
                    groups[group].append(allele)
        
        # Calculate intra-group similarities
        intra_group_similarities = []
        inter_group_similarities = []
        
        # Measure intra-group similarities
        for group, alleles in groups.items():
            if len(alleles) < 2:
                continue
                
            # Calculate all pairwise similarities within group
            group_similarities = []
            for i, allele1 in enumerate(alleles):
                for allele2 in alleles[i+1:]:
                    similarity = self.encoder._cosine_similarity(
                        embeddings[allele1],
                        embeddings[allele2]
                    )
                    group_similarities.append(similarity)
                    
            if group_similarities:
                intra_group_similarities.append((group, np.mean(group_similarities)))
        
        # Measure inter-group similarities
        group_list = list(groups.keys())
        for i, group1 in enumerate(group_list):
            for group2 in group_list[i+1:]:
                # Skip groups without enough alleles
                if len(groups[group1]) < 1 or len(groups[group2]) < 1:
                    continue
                    
                # Calculate pairwise similarities between groups
                # (using first allele from each group for efficiency)
                allele1 = groups[group1][0]
                allele2 = groups[group2][0]
                
                similarity = self.encoder._cosine_similarity(
                    embeddings[allele1],
                    embeddings[allele2]
                )
                inter_group_similarities.append(((group1, group2), similarity))
        
        # Calculate metrics
        intra_similarities = [s for _, s in intra_group_similarities]
        inter_similarities = [s for _, s in inter_group_similarities]
        
        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0
        avg_inter_similarity = np.mean(inter_similarities) if inter_similarities else 0
        
        # Calculate separation (difference between intra and inter similarities)
        separation = avg_intra_similarity - avg_inter_similarity
        
        # Prepare results
        results = {
            "num_groups": len(groups),
            "total_alleles": len(embeddings),
            "avg_intra_group_similarity": avg_intra_similarity,
            "avg_inter_group_similarity": avg_inter_similarity,
            "group_separation": separation,
            "intra_group_details": intra_group_similarities,
            "inter_group_details": inter_group_similarities
        }
        
        return results
    
    def evaluate_recovery(self, locus: str = None, samples: int = 50) -> Dict:
        """Evaluate nearest neighbor recovery accuracy
        
        This evaluates whether embeddings allow recovery of the
        correct allele group from a random allele.
        
        Args:
            locus: HLA locus to evaluate (or None for all loci)
            samples: Number of random samples to test
            
        Returns:
            Dict with recovery metrics
        """
        # Get embeddings for the locus (or all)
        if locus:
            embeddings = {
                allele: emb for allele, emb in self.encoder.embeddings.items()
                if allele.startswith(f"{locus}*")
            }
        else:
            embeddings = self.encoder.embeddings
            
        if not embeddings:
            logger.error(f"No embeddings found for evaluation")
            return {"error": "No embeddings found"}
            
        # Group alleles by first field
        allele_groups = {}
        for allele in embeddings:
            if '*' in allele:
                parts = allele.split('*')
                if len(parts) > 1 and ':' in parts[1]:
                    group = f"{parts[0]}*{parts[1].split(':')[0]}"
                    allele_groups[allele] = group
        
        # Run recovery tests
        all_alleles = list(embeddings.keys())
        recovery_results = []
        
        for _ in range(min(samples, len(all_alleles))):
            # Select random query allele
            query_idx = np.random.randint(0, len(all_alleles))
            query_allele = all_alleles[query_idx]
            
            # Skip if we can't determine its group
            if query_allele not in allele_groups:
                continue
                
            query_group = allele_groups[query_allele]
            
            # Find nearest neighbors
            similar_alleles = self.encoder.find_similar_alleles(
                query_allele, top_k=5, metric="cosine"
            )
            
            # Check if neighbors are from the same group
            correct_group = 0
            for neighbor_allele, _ in similar_alleles:
                if neighbor_allele in allele_groups:
                    neighbor_group = allele_groups[neighbor_allele]
                    if neighbor_group == query_group:
                        correct_group += 1
            
            recovery = correct_group / len(similar_alleles) if similar_alleles else 0
            recovery_results.append((query_allele, recovery))
        
        # Calculate metrics
        recoveries = [r for _, r in recovery_results]
        avg_recovery = np.mean(recoveries) if recoveries else 0
        
        # Prepare results
        results = {
            "avg_recovery": avg_recovery,
            "num_samples": len(recovery_results),
            "recovery_details": recovery_results
        }
        
        return results
    
    def plot_consistency_results(
        self,
        results: Dict,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot consistency evaluation results
        
        Args:
            results: Results from evaluate_consistency
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; visualization not available")
            return None
            
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Intra-group vs Inter-group similarities
        ax = axes[0]
        
        data = pd.DataFrame({
            'Type': ['Intra-group', 'Inter-group'],
            'Average Similarity': [
                results['avg_intra_group_similarity'],
                results['avg_inter_group_similarity']
            ]
        })
        
        sns.barplot(x='Type', y='Average Similarity', data=data, ax=ax)
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Average Similarities')
        ax.set_ylim(0, 1)
        
        # Add values on bars
        for i, v in enumerate(data['Average Similarity']):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        # Add separation value
        ax.text(0.5, 0.9, f"Separation: {results['group_separation']:.3f}", 
               ha='center', transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.8))
        
        # 2. Top intra-group similarities
        ax = axes[1]
        
        # Sort by similarity (descending)
        top_groups = sorted(
            results['intra_group_details'], 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Show top 10
        
        if top_groups:
            group_data = pd.DataFrame({
                'Group': [g for g, _ in top_groups],
                'Similarity': [s for _, s in top_groups]
            })
            
            sns.barplot(x='Similarity', y='Group', data=group_data, ax=ax)
            ax.set_title('Top Groups by Intra-similarity')
            ax.set_xlim(0, 1)
            
            # Add values on bars
            for i, v in enumerate(group_data['Similarity']):
                ax.text(v + 0.02, i, f"{v:.3f}", va='center')
        else:
            ax.text(0.5, 0.5, "No group data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
        
    def plot_recovery_results(
        self,
        results: Dict,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot recovery evaluation results
        
        Args:
            results: Results from evaluate_recovery
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; visualization not available")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract recoveries
        recoveries = [r for _, r in results['recovery_details']]
        
        # Plot histogram
        sns.histplot(recoveries, bins=5, kde=True, ax=ax)
        ax.set_xlabel('Recovery Rate')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Nearest Neighbor Recovery Distribution (Avg: {results["avg_recovery"]:.3f})')
        
        # Add average line
        ax.axvline(results['avg_recovery'], color='red', linestyle='--', 
                  label=f'Average: {results["avg_recovery"]:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig

class PredictionEvaluator:
    """Evaluates the performance of HLA-based prediction models"""
    
    def evaluate_classification(
        self,
        y_true: List[int],
        y_pred: List[float],
        threshold: float = 0.5
    ) -> Dict:
        """Evaluate binary classification performance
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            threshold: Decision threshold for binary classification
            
        Returns:
            Dict with classification metrics
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_true, y_pred_binary)
        precision = metrics.precision_score(y_true, y_pred_binary)
        recall = metrics.recall_score(y_true, y_pred_binary)
        f1 = metrics.f1_score(y_true, y_pred_binary)
        
        # ROC and AUC
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        
        # Precision-Recall
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_pred)
        avg_precision = metrics.average_precision_score(y_true, y_pred)
        
        # Confusion matrix
        confusion = metrics.confusion_matrix(y_true, y_pred_binary)
        
        # Prepare results
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "average_precision": avg_precision,
            "confusion_matrix": confusion,
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr
            },
            "pr_curve": {
                "precision": precision_curve,
                "recall": recall_curve
            }
        }
        
        return results
    
    def evaluate_regression(
        self,
        y_true: List[float],
        y_pred: List[float]
    ) -> Dict:
        """Evaluate regression performance
        
        Args:
            y_true: True continuous values
            y_pred: Predicted values
            
        Returns:
            Dict with regression metrics
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        
        # Calculate explained variance
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        
        # Prepare results
        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "explained_variance": explained_variance
        }
        
        return results
    
    def plot_classification_results(
        self,
        results: Dict,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot classification evaluation results
        
        Args:
            results: Results from evaluate_classification
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; visualization not available")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. ROC Curve
        ax = axes[0, 0]
        ax.plot(
            results['roc_curve']['fpr'], 
            results['roc_curve']['tpr'], 
            lw=2, 
            label=f"AUC = {results['auc']:.3f}"
        )
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        
        # 2. Precision-Recall Curve
        ax = axes[0, 1]
        ax.plot(
            results['pr_curve']['recall'], 
            results['pr_curve']['precision'], 
            lw=2,
            label=f"AP = {results['average_precision']:.3f}"
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        # 3. Confusion Matrix
        ax = axes[1, 0]
        cm = results['confusion_matrix']
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        # 4. Metrics Summary
        ax = axes[1, 1]
        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
            'Value': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score'],
                results['auc']
            ]
        })
        
        # Create a table-like bar chart
        bars = ax.barh(metrics_data['Metric'], metrics_data['Value'])
        ax.set_xlim([0, 1])
        ax.set_xlabel('Value')
        ax.set_title('Performance Metrics')
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", 
                va='center'
            )
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def plot_regression_results(
        self,
        results: Dict,
        y_true: List[float],
        y_pred: List[float],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot regression evaluation results
        
        Args:
            results: Results from evaluate_regression
            y_true: True continuous values
            y_pred: Predicted values
            output_file: Path to save visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; visualization not available")
            return None
            
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Scatter plot of true vs predicted
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'True vs Predicted (R² = {results["r2"]:.3f})')
        
        # Add R2 text
        ax.text(
            0.05, 
            0.95, 
            f"R² = {results['r2']:.3f}", 
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # 2. Metrics Summary
        ax = axes[1]
        metrics_data = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Expl. Var.'],
            'Value': [
                results['mse'],
                results['rmse'],
                results['mae'],
                results['r2'],
                results['explained_variance']
            ]
        })
        
        # Create a table-like bar chart
        bars = ax.barh(metrics_data['Metric'], metrics_data['Value'])
        ax.set_xlabel('Value')
        ax.set_title('Performance Metrics')
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", 
                va='center'
            )
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
