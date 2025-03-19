#!/usr/bin/env python
"""
HLA Locus-Specific Embedding Analysis
-------------------------------------
Generate and visualize embeddings for specific HLA loci using different dimensionality
reduction techniques (UMAP, t-SNE, and PCA).
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our modules
from src.models.protbert import ProtBERTEncoder
from src.analysis.visualization import HLAEmbeddingVisualizer
from src.utils.logging import setup_logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate and visualize HLA locus-specific embeddings"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./data/analysis/locus_embeddings",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--loci", 
        type=str, 
        nargs="+",
        default=["A", "B", "C", "DRB1", "DQB1", "DPB1"],
        help="HLA loci to analyze"
    )
    
    parser.add_argument(
        "--sequence-file", 
        type=str, 
        default="./data/processed/hla_sequences.pkl",
        help="Path to HLA sequence file"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="./data/embeddings",
        help="Directory to cache embeddings"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for encoding"
    )
    
    parser.add_argument(
        "--min-alleles",
        type=int,
        default=5,
        help="Minimum number of alleles required for a locus to be analyzed"
    )
    
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="n_neighbors parameter for UMAP"
    )
    
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="min_dist parameter for UMAP"
    )
    
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE"
    )
    
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="Learning rate parameter for t-SNE"
    )
    
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of components for PCA"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def make_output_dirs(base_dir: str) -> Tuple[Path, Path, Path]:
    """Create output directories for embeddings and plots"""
    base_path = Path(base_dir)
    
    # Create directories
    embeddings_dir = base_path / "embeddings"
    plots_dir = base_path / "plots"
    reports_dir = base_path / "reports"
    
    embeddings_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    return embeddings_dir, plots_dir, reports_dir

def process_locus(
    locus: str,
    encoder: ProtBERTEncoder,
    visualizer: HLAEmbeddingVisualizer,
    embeddings_dir: Path,
    plots_dir: Path,
    batch_size: int = 8,
    min_alleles: int = 5,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float = 200.0,
    pca_components: int = 2
) -> Dict:
    """Process a specific HLA locus"""
    start_time = time.time()
    logger = logging.getLogger("locus_analysis")
    
    # Get alleles for this locus
    alleles = [
        allele for allele in encoder.sequences.keys()
        if allele.startswith(f"{locus}*")
    ]
    
    if len(alleles) < min_alleles:
        logger.warning(f"Skipping locus {locus}: Only {len(alleles)} alleles found (minimum {min_alleles})")
        return {
            "locus": locus,
            "status": "skipped",
            "reason": f"Only {len(alleles)} alleles found (minimum {min_alleles})",
            "allele_count": len(alleles)
        }
    
    logger.info(f"Processing locus {locus} with {len(alleles)} alleles")
    
    # Get sequences
    logger.info(f"Getting sequences for {len(alleles)} {locus} alleles")
    sequences = []
    valid_alleles = []
    
    for allele in tqdm(alleles, desc=f"Getting {locus} sequences"):
        sequence = encoder.get_sequence(allele)
        if sequence is not None:
            sequences.append(sequence)
            valid_alleles.append(allele)
    
    logger.info(f"Found {len(valid_alleles)} valid sequences for locus {locus}")
    
    # Get embeddings in batches
    logger.info(f"Encoding {len(valid_alleles)} {locus} alleles")
    embeddings = {}
    
    # Process in batches
    for i in tqdm(range(0, len(valid_alleles), batch_size), desc=f"Encoding {locus} alleles"):
        batch_alleles = valid_alleles[i:i + batch_size]
        batch_embeddings = encoder.batch_encode_alleles(batch_alleles)
        embeddings.update(batch_embeddings)
    
    # Save embeddings
    embeddings_file = embeddings_dir / f"hla_{locus}_embeddings.pkl"
    pd.to_pickle(embeddings, embeddings_file)
    logger.info(f"Saved {len(embeddings)} {locus} embeddings to {embeddings_file}")
    
    # Create visualizations
    
    # 1. UMAP visualization
    logger.info(f"Generating UMAP visualization for {locus}")
    umap_file = plots_dir / f"hla_{locus}_umap.png"
    visualizer.visualize_embeddings(
        embeddings,
        method="umap",
        color_by="group",
        output_file=str(umap_file),
        title=f"HLA-{locus} Embeddings - UMAP Projection",
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=42
    )
    
    # 2. t-SNE visualization
    logger.info(f"Generating t-SNE visualization for {locus}")
    tsne_file = plots_dir / f"hla_{locus}_tsne.png"
    visualizer.visualize_embeddings(
        embeddings,
        method="tsne",
        color_by="group",
        output_file=str(tsne_file),
        title=f"HLA-{locus} Embeddings - t-SNE Projection",
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        random_state=42
    )
    
    # 3. PCA visualization
    logger.info(f"Generating PCA visualization for {locus}")
    pca_file = plots_dir / f"hla_{locus}_pca.png"
    visualizer.visualize_embeddings(
        embeddings,
        method="pca",
        color_by="group",
        output_file=str(pca_file),
        title=f"HLA-{locus} Embeddings - PCA Projection",
        n_components=pca_components
    )
    
    # 4. Allele groups visualization
    logger.info(f"Generating allele groups visualization for {locus}")
    groups_file = plots_dir / f"hla_{locus}_groups.png"
    visualizer.visualize_allele_groups(
        locus,
        method="umap",
        min_alleles_per_group=5,
        output_file=str(groups_file),
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=42
    )
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Return stats
    return {
        "locus": locus,
        "status": "success",
        "allele_count": len(valid_alleles),
        "embedding_count": len(embeddings),
        "processing_time": processing_time,
        "plots": {
            "umap": str(umap_file),
            "tsne": str(tsne_file),
            "pca": str(pca_file),
            "groups": str(groups_file),
        }
    }

def write_report(results: List[Dict], reports_dir: Path) -> str:
    """Generate a summary report of the analysis"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = reports_dir / "locus_analysis_report.md"
    
    with open(report_file, "w") as f:
        f.write(f"# HLA Locus Embedding Analysis Report\n\n")
        f.write(f"Generated: {now}\n\n")
        
        f.write("## Summary\n\n")
        total_alleles = sum(result.get("allele_count", 0) for result in results)
        total_embeddings = sum(result.get("embedding_count", 0) for result in results 
                             if result.get("status") == "success")
        total_time = sum(result.get("processing_time", 0) for result in results
                       if result.get("status") == "success")
        
        f.write(f"- Total HLA loci analyzed: {len(results)}\n")
        f.write(f"- Total alleles processed: {total_alleles}\n")
        f.write(f"- Total embeddings generated: {total_embeddings}\n")
        f.write(f"- Total processing time: {total_time:.2f} seconds\n\n")
        
        f.write("## Loci Analysis\n\n")
        
        for result in results:
            locus = result["locus"]
            status = result["status"]
            
            f.write(f"### HLA-{locus}\n\n")
            f.write(f"- Status: {status}\n")
            
            if status == "success":
                f.write(f"- Allele count: {result['allele_count']}\n")
                f.write(f"- Embedding count: {result['embedding_count']}\n")
                f.write(f"- Processing time: {result['processing_time']:.2f} seconds\n\n")
                
                f.write("#### Visualizations\n\n")
                for plot_type, plot_path in result["plots"].items():
                    f.write(f"- {plot_type.upper()}: [{plot_type}]({os.path.relpath(plot_path, reports_dir.parent)})\n")
                f.write("\n")
            else:
                f.write(f"- Reason: {result['reason']}\n\n")
    
    return str(report_file)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)
    logger.info("Starting HLA locus-specific embedding analysis")
    
    # Create output directories
    embeddings_dir, plots_dir, reports_dir = make_output_dirs(args.output_dir)
    logger.info(f"Output directories created at {args.output_dir}")
    
    # Initialize encoder
    logger.info(f"Initializing ProtBERTEncoder")
    encoder = ProtBERTEncoder(
        sequence_file=args.sequence_file,
        cache_dir=args.cache_dir
    )
    
    # Initialize visualizer
    visualizer = HLAEmbeddingVisualizer(encoder)
    
    # Process each locus
    results = []
    for locus in args.loci:
        try:
            logger.info(f"Processing locus {locus}")
            result = process_locus(
                locus=locus,
                encoder=encoder,
                visualizer=visualizer,
                embeddings_dir=embeddings_dir,
                plots_dir=plots_dir,
                batch_size=args.batch_size,
                min_alleles=args.min_alleles,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                tsne_perplexity=args.tsne_perplexity,
                tsne_learning_rate=args.tsne_learning_rate,
                pca_components=args.pca_components
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing locus {locus}: {e}")
            results.append({
                "locus": locus,
                "status": "error",
                "reason": str(e),
                "allele_count": 0
            })
    
    # Generate report
    report_file = write_report(results, reports_dir)
    logger.info(f"Analysis report written to {report_file}")
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"Analysis complete: {success_count} loci processed successfully, "
                f"{skipped_count} skipped, {error_count} errors")
    
    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())
