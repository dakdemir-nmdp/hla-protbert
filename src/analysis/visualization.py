"""
HLA Embedding Visualization
-------------------------
Tools for visualizing HLA embeddings using dimensionality reduction.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check for visualization packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed; visualization features will be limited")
    PLOTTING_AVAILABLE = False

# Check for dimensionality reduction packages
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP
    DR_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn or UMAP not installed; dimensionality reduction will be limited")
    DR_AVAILABLE = False

class HLAEmbeddingVisualizer:
    """Visualizes HLA embeddings using dimensionality reduction techniques"""
    
    def __init__(self, encoder):
        """Initialize visualizer
        
        Args:
            encoder: HLAEncoder instance
        """
        self.encoder = encoder
        
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib/Seaborn not installed; visualization not available")
            
        if not DR_AVAILABLE:
            logger.error("scikit-learn or UMAP not installed; dimensionality reduction not available")

    def visualize_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str = 'pca',
        color_by: str = 'locus',
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        **kwargs
    ):
        """Simplified version - just support key methods"""
        if not PLOTTING_AVAILABLE or not DR_AVAILABLE:
            return None
        
        # Get allele names and embeddings as array
        allele_names = list(embeddings.keys())
        X = np.array([embeddings[allele] for allele in allele_names])
        
        # Do dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, **kwargs)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, **kwargs)
        elif method == 'umap':
            reducer = UMAP(n_components=2, **kwargs)
        else:
            print(f"Unknown method: {method}, using PCA")
            reducer = PCA(n_components=2)
            
        reduced = reducer.fit_transform(X)
        
        # Extract metadata from allele names
        metadata = []
        for allele in allele_names:
            if '*' in allele:
                locus, fields = allele.split('*', 1)
                fields = fields.split(':')
                group = fields[0] if len(fields) > 0 else ''
            else:
                locus = allele
                group = ''
                
            metadata.append({
                'allele': allele,
                'locus': locus,
                'group': group,
                'class': 'I' if locus in ['A', 'B', 'C'] else 'II'
            })
            
        metadata = pd.DataFrame(metadata)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        unique_values = metadata[color_by].unique()
        
        # Create a color map
        if len(unique_values) <= 10:
            palette = sns.color_palette("tab10", len(unique_values))
        else:
            palette = sns.color_palette("husl", len(unique_values))
            
        # Plot each value with different color
        for i, val in enumerate(sorted(unique_values)):
            mask = metadata[color_by] == val
            points = reduced[mask]
            if len(points) > 0:
                plt.scatter(
                    points[:, 0], 
                    points[:, 1], 
                    c=[palette[i]], 
                    label=val,
                    alpha=0.7
                )
                
        # Add legend and labels
        plt.legend(title=color_by)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"HLA Embeddings - {method.upper()} colored by {color_by}")
            
        plt.grid(alpha=0.3)
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300)
            return None
        else:
            return plt
            
    def visualize_allele_groups(self, locus, **kwargs):
        """Stub implementation to prevent errors"""
        return self.visualize_embeddings(
            self.encoder.embeddings,
            color_by='group',
            title=f"HLA-{locus} Allele Groups"
        )