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
    from sklearn.manifold import TSNE, MDS
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
    
    def reduce_dimensions(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str = 'umap',
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """Reduce dimensionality of embeddings
        
        Args:
            embeddings: Dict mapping allele names to embeddings
            method: Dimensionality reduction method ('pca', 'tsne', 'umap', or 'mds')
            n_components: Number of dimensions to reduce to
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Tuple of (reduced_embeddings, allele_names)
        """
        if not DR_AVAILABLE:
            raise ImportError("scikit-learn or UMAP not installed; dimensionality reduction not available")
            
        # Extract embeddings and allele names
        allele_names = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[allele] for allele in allele_names])
        
        # Choose reduction method
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, **kwargs)
        elif method.lower() == 'umap':
            reducer = UMAP(n_components=n_components, **kwargs)
        elif method.lower() == 'mds':
            reducer = MDS(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        # Perform reduction
        reduced = reducer.fit_transform(embedding_matrix)
        
        return reduced, allele_names
    
    def get_allele_metadata(self, allele_names: List[str]) -> pd.DataFrame:
        """Extract metadata from allele names
        
        Args:
            allele_names: List of HLA allele names
            
        Returns:
            DataFrame with allele metadata
        """
        # Extract metadata from allele names
        metadata = []
        
        for allele in allele_names:
            # Extract basic info
            if '*' in allele:
                locus, fields = allele.split('*', 1)
                fields = fields.split(':')
                group = fields[0] if len(fields) > 0 else ''
                protein = fields[1] if len(fields) > 1 else ''
            else:
                locus = allele
                group = ''
                protein = ''
                
            # Add to metadata
            metadata.append({
                'allele': allele,
                'locus': locus,
                'group': group,
                'protein': protein,
                'class': 'I' if locus in ['A', 'B', 'C'] else 'II'
            })
        
        return pd.DataFrame(metadata)
    
    def visualize_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str = 'umap',
        color_by: str = 'locus',
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        **kwargs
    ):
        """Visualize embeddings using dimensionality reduction
        
        Args:
            embeddings: Dict mapping allele names to embeddings
            method: Dimensionality reduction method ('pca', 'tsne', 'umap', or 'mds')
            color_by: Metadata column to use for coloring ('locus', 'group', 'class')
            output_file: Path to save visualization
            figsize: Figure size
            title: Plot title
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE or not DR_AVAILABLE:
            logger.error("Required packages not installed; visualization not available")
            return None
            
        # Reduce dimensions
        reduced, allele_names = self.reduce_dimensions(embeddings, method, **kwargs)
        
        # Get metadata
        metadata = self.get_allele_metadata(allele_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine color palette based on unique values
        unique_values = metadata[color_by].unique()
        if len(unique_values) <= 10:
            palette = sns.color_palette("tab10", len(unique_values))
        else:
            palette = sns.color_palette("husl", len(unique_values))
            
        # Create colormap
        color_map = {val: palette[i] for i, val in enumerate(sorted(unique_values))}
        
        # Plot points colored by metadata
        for val in unique_values:
            mask = metadata[color_by] == val
            points = reduced[mask]
            ax.scatter(
                points[:, 0], 
                points[:, 1], 
                c=[color_map[val]] * sum(mask), 
                label=val,
                alpha=0.7,
                edgecolors='none'
            )
            
        # Enhance plot
        ax.legend(title=color_by)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"HLA Embeddings - {method.upper()} projection colored by {color_by}")
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add tight layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def visualize_locus(
        self,
        locus: str,
        method: str = 'umap',
        color_by: str = 'group',
        annotate: bool = True,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        **kwargs
    ):
        """Visualize embeddings for a specific HLA locus
        
        Args:
            locus: HLA locus to visualize (e.g., 'A', 'B', 'DRB1')
            method: Dimensionality reduction method ('pca', 'tsne', 'umap', or 'mds')
            color_by: Metadata column to use for coloring
            annotate: Whether to annotate points with allele names
            output_file: Path to save visualization
            figsize: Figure size
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE or not DR_AVAILABLE:
            logger.error("Required packages not installed; visualization not available")
            return None
            
        # Get embeddings for the locus
        locus_embeddings = {}
        for allele, emb in self.encoder.embeddings.items():
            if allele.startswith(f"{locus}*"):
                locus_embeddings[allele] = emb
        
        if not locus_embeddings:
            logger.error(f"No embeddings found for locus {locus}")
            return None
            
        # Reduce dimensions
        reduced, allele_names = self.reduce_dimensions(locus_embeddings, method, **kwargs)
        
        # Get metadata
        metadata = self.get_allele_metadata(allele_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine color palette based on unique values
        unique_values = metadata[color_by].unique()
        if len(unique_values) <= 10:
            palette = sns.color_palette("tab10", len(unique_values))
        else:
            palette = sns.color_palette("husl", len(unique_values))
            
        # Create colormap
        color_map = {val: palette[i] for i, val in enumerate(sorted(unique_values))}
        
        # Plot points colored by metadata
        for val in unique_values:
            mask = metadata[color_by] == val
            points = reduced[mask]
            ax.scatter(
                points[:, 0], 
                points[:, 1], 
                c=[color_map[val]] * sum(mask), 
                label=val,
                alpha=0.7,
                s=100,
                edgecolors='w',
                linewidth=0.5
            )
        
        # Annotate points
        if annotate:
            for i, allele in enumerate(allele_names):
                ax.annotate(
                    allele.split('*')[1],  # Skip locus prefix for cleaner labels
                    (reduced[i, 0], reduced[i, 1]),
                    fontsize=8,
                    alpha=0.7,
                    ha='center',
                    va='center',
                    xytext=(0, 5),
                    textcoords='offset points'
                )
                
        # Enhance plot
        ax.legend(title=color_by)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"HLA-{locus} Embeddings - {method.upper()} projection")
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add tight layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def plot_similarity_heatmap(
        self,
        alleles: List[str],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'viridis',
        annotate: bool = True
    ):
        """Plot similarity heatmap between alleles
        
        Args:
            alleles: List of HLA alleles to compare
            output_file: Path to save visualization
            figsize: Figure size
            cmap: Colormap name
            annotate: Whether to annotate cells with similarity values
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not installed; visualization not available")
            return None
            
        # Calculate pairwise similarities
        n = len(alleles)
        similarity_matrix = np.zeros((n, n))
        
        # Get embeddings
        embeddings = {}
        for allele in alleles:
            try:
                embeddings[allele] = self.encoder.get_embedding(allele)
            except Exception as e:
                logger.error(f"Error getting embedding for {allele}: {e}")
                return None
        
        # Calculate similarities
        for i, allele1 in enumerate(alleles):
            for j, allele2 in enumerate(alleles):
                similarity = self.encoder._cosine_similarity(
                    embeddings[allele1],
                    embeddings[allele2]
                )
                similarity_matrix[i, j] = similarity
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(alleles, rotation=45, ha='right')
        ax.set_yticklabels(alleles)
        
        # Add annotations
        if annotate:
            for i in range(n):
                for j in range(n):
                    text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                                  ha="center", va="center", 
                                  color="w" if similarity_matrix[i, j] < 0.7 else "black")
        
        # Set title
        ax.set_title("HLA Allele Similarity Matrix")
        
        # Add tight layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def visualize_allele_groups(
        self,
        locus: str,
        method: str = 'umap',
        min_alleles_per_group: int = 5,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        **kwargs
    ):
        """Visualize embedding clusters by allele groups
        
        Args:
            locus: HLA locus to visualize (e.g., 'A', 'B', 'DRB1')
            method: Dimensionality reduction method
            min_alleles_per_group: Minimum number of alleles per group to include
            output_file: Path to save visualization
            figsize: Figure size
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Matplotlib figure if output_file is None, otherwise None
        """
        if not PLOTTING_AVAILABLE or not DR_AVAILABLE:
            logger.error("Required packages not installed; visualization not available")
            return None
            
        # Get embeddings for the locus
        locus_embeddings = {}
        for allele, emb in self.encoder.embeddings.items():
            if allele.startswith(f"{locus}*"):
                locus_embeddings[allele] = emb
        
        if not locus_embeddings:
            logger.error(f"No embeddings found for locus {locus}")
            return None
            
        # Get metadata
        metadata = self.get_allele_metadata(list(locus_embeddings.keys()))
        
        # Count alleles per group
        group_counts = metadata['group'].value_counts()
        
        # Filter groups with enough alleles
        valid_groups = group_counts[group_counts >= min_alleles_per_group].index.tolist()
        
        # Filter embeddings
        filtered_embeddings = {}
        for allele, emb in locus_embeddings.items():
            if '*' in allele:
                group = allele.split('*')[1].split(':')[0]
                if group in valid_groups:
                    filtered_embeddings[allele] = emb
        
        if not filtered_embeddings:
            logger.error(f"No groups with at least {min_alleles_per_group} alleles found for locus {locus}")
            return None
            
        # Reduce dimensions
        reduced, allele_names = self.reduce_dimensions(filtered_embeddings, method, **kwargs)
        
        # Get metadata for filtered alleles
        metadata = self.get_allele_metadata(allele_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine color palette
        palette = sns.color_palette("husl", len(valid_groups))
            
        # Create colormap
        color_map = {val: palette[i] for i, val in enumerate(sorted(valid_groups))}
        
        # Plot points colored by group
        for group in valid_groups:
            mask = metadata['group'] == group
            points = reduced[mask]
            
            # Calculate centroid
            centroid = points.mean(axis=0)
            
            # Plot points
            ax.scatter(
                points[:, 0], 
                points[:, 1], 
                c=[color_map[group]] * sum(mask), 
                label=f"{locus}*{group}",
                alpha=0.7,
                s=100,
                edgecolors='w',
                linewidth=0.5
            )
            
            # Plot centroid and label
            ax.scatter(
                centroid[0],
                centroid[1],
                c=[color_map[group]],
                s=200,
                marker='*',
                edgecolors='k',
                linewidth=1.5
            )
            
            ax.text(
                centroid[0],
                centroid[1],
                f"{locus}*{group}",
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
                
        # Enhance plot
        ax.legend(title="Allele Groups", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"HLA-{locus} Allele Groups - {method.upper()} projection")
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add tight layout
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
