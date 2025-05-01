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
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

# Import our modules
try:
    from src.models.encoders import ProtBERTEncoder, ESMEncoder # Import both
    from src.models.encoder import HLAEncoder # Import base class for type hinting
    from src.analysis.visualization import HLAEmbeddingVisualizer
    from src.utils.logging import setup_logging
    from src.utils.config import ConfigManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running the script from the project root directory.")
    sys.exit(1)

# Check for required packages
try:
    import umap
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as e:
    print(f"Required package not installed: {e}")
    print("Please install required packages with: pip install scikit-learn umap-learn")
    sys.exit(1)

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
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (extra checks and exception details)"
    )

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
        help="Pooling strategy for ESM embeddings ('mean' or 'cls')"
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

def check_environment(args):
    """Check environment and file paths before starting analysis"""
    logger = logging.getLogger("environment_check")
    success = True
    
    # Check sequence file
    sequence_file = Path(args.sequence_file)
    if not sequence_file.exists():
        logger.error(f"Sequence file not found: {sequence_file}")
        success = False
    else:
        logger.info(f"Sequence file found: {sequence_file}")
        
    # Check cache directory
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        try:
            cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            success = False
    else:
        logger.info(f"Cache directory found: {cache_dir}")
    
    # Check output directory permissions
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not os.access(output_dir, os.W_OK):
            logger.error(f"No write permission for output directory: {output_dir}")
            success = False
    else:
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            success = False
    
    return success

def process_locus(
    locus: str,
    encoder: HLAEncoder, # Use base class for type hint
    visualizer: HLAEmbeddingVisualizer,
    embeddings_dir: Path,
    plots_dir: Path,
    batch_size: int = 8,
    min_alleles: int = 5,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float = 200.0,
    pca_components: int = 2,
    debug: bool = False
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
    
    if len(valid_alleles) == 0:
        logger.error(f"No valid sequences found for locus {locus}")
        return {
            "locus": locus,
            "status": "error",
            "reason": "No valid sequences found",
            "allele_count": 0
        }
    
    # Get embeddings in batches
    logger.info(f"Encoding {len(valid_alleles)} {locus} alleles")
    embeddings = {}
    
    try:
        # First check which alleles are already cached in the encoder
        cached_alleles = []
        uncached_alleles = []
        
        for allele in valid_alleles:
            if allele in encoder.embeddings:
                cached_alleles.append(allele)
                embeddings[allele] = encoder.embeddings[allele]
            else:
                uncached_alleles.append(allele)
        
        logger.info(f"Found {len(cached_alleles)} cached embeddings, {len(uncached_alleles)} to encode")
        
        # Process uncached alleles in batches
        for i in tqdm(range(0, len(uncached_alleles), batch_size), desc=f"Encoding {locus} alleles"):
            batch_alleles = uncached_alleles[i:i + batch_size]
            batch_embeddings = encoder.batch_encode_alleles(batch_alleles)
            embeddings.update(batch_embeddings)
    
        # Double-check dimensions
        embedding_dims = [emb.shape[0] for emb in embeddings.values()]
        if len(set(embedding_dims)) > 1:
            logger.warning(f"Inconsistent embedding dimensions found: {set(embedding_dims)}")
            
            # Use the most common dimension
            from collections import Counter
            most_common_dim = Counter(embedding_dims).most_common(1)[0][0]
            logger.info(f"Using most common dimension: {most_common_dim}")
            
            # Filter embeddings to keep only those with the most common dimension
            embeddings = {
                allele: emb for allele, emb in embeddings.items() 
                if emb.shape[0] == most_common_dim
            }
            logger.info(f"Filtered to {len(embeddings)} embeddings with consistent dimensions")
    except Exception as e:
        if debug:
            logger.error(f"Error encoding alleles: {e}\n{traceback.format_exc()}")
        else:
            logger.error(f"Error encoding alleles: {e}")
        return {
            "locus": locus,
            "status": "error",
            "reason": f"Encoding error: {str(e)}",
            "allele_count": len(valid_alleles)
        }
    
    # Save embeddings
    embeddings_file = embeddings_dir / f"hla_{locus}_embeddings.pkl"
    try:
        pd.to_pickle(embeddings, embeddings_file)
        logger.info(f"Saved {len(embeddings)} {locus} embeddings to {embeddings_file}")
    except Exception as e:
        logger.error(f"Error saving embeddings to {embeddings_file}: {e}")
        if debug:
            logger.error(traceback.format_exc())
    
    # Create visualizations
    plot_paths = {}
    vis_methods = [
        ("umap", {
            "n_neighbors": umap_neighbors,
            "min_dist": umap_min_dist,
            "random_state": 42
        }),
        ("tsne", {
            "perplexity": tsne_perplexity,
            "learning_rate": tsne_learning_rate,
            "random_state": 42
        }),
        ("pca", {
            "n_components": pca_components
        })
    ]
    
    for method, params in vis_methods:
        try:
            logger.info(f"Generating {method.upper()} visualization for {locus}")
            output_file = plots_dir / f"hla_{locus}_{method}.png"
            
            visualizer.visualize_embeddings(
                embeddings,
                method=method,
                color_by="group",
                output_file=str(output_file),
                title=f"HLA-{locus} Embeddings - {method.upper()} Projection",
                **params
            )
            
            plot_paths[method] = str(output_file)
        except Exception as e:
            logger.error(f"Error generating {method} visualization: {e}")
            if debug:
                logger.error(traceback.format_exc())
            plot_paths[method] = "error"
    
    # Generate allele groups visualization
    try:
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
        
        plot_paths["groups"] = str(groups_file)
    except Exception as e:
        logger.error(f"Error generating groups visualization: {e}")
        if debug:
            logger.error(traceback.format_exc())
        plot_paths["groups"] = "error"
    
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
        "plots": plot_paths
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
        
        # Success, skipped, and error counts
        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        error_count = sum(1 for r in results if r["status"] == "error")
        
        f.write(f"- Successfully processed: {success_count} loci\n")
        f.write(f"- Skipped: {skipped_count} loci\n")
        f.write(f"- Errors: {error_count} loci\n\n")
        
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
                    if plot_path == "error":
                        f.write(f"- {plot_type.upper()}: Failed to generate\n")
                    else:
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
    log_level = logging.DEBUG if args.verbose or args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    logger.info("Starting HLA locus-specific embedding analysis")
    
    # Check environment
    if not check_environment(args):
        logger.error("Environment check failed. Fix the issues above and try again.")
        return 1
    
    # Create output directories
    embeddings_dir, plots_dir, reports_dir = make_output_dirs(args.output_dir)
    logger.info(f"Output directories created at {args.output_dir}")
    
    # Initialize encoder based on type
    try:
        logger.info(f"Initializing {args.encoder_type.upper()} encoder")
        encoder_args = {
            "sequence_file": args.sequence_file,
            "cache_dir": args.cache_dir,
            "device": args.device,
            # Add model_name if provided, otherwise let class use default
            **({"model_name": args.model_name} if args.model_name else {})
        }

        if args.encoder_type == "protbert":
            EncoderClass = ProtBERTEncoder
            # ProtBERT specific args if any (e.g., model_name default)
            if not args.model_name:
                 encoder_args["model_name"] = "Rostlab/prot_bert" # Default for ProtBERT
        elif args.encoder_type == "esm":
            EncoderClass = ESMEncoder
            # ESM specific args
            encoder_args["pooling_strategy"] = args.pooling_strategy
            # Add HF token if available in config
            config_manager = ConfigManager()
            hf_token = config_manager.get("model.hf_token", None)
            if hf_token:
                encoder_args["hf_token"] = hf_token
            if not args.model_name:
                 encoder_args["model_name"] = "facebook/esm2_t33_650M_UR50D" # Default for ESM
        else:
            # This case should not be reached due to argparse choices
            raise ValueError(f"Unsupported encoder type: {args.encoder_type}")

        encoder = EncoderClass(**encoder_args)
        
        logger.info(f"{args.encoder_type.upper()} encoder initialized with {len(encoder.sequences)} sequences and {len(encoder.embeddings)} cached embeddings")
        logger.info(f"Using model: {encoder.model_name} on device: {encoder.device}")
        if args.encoder_type == "esm":
             logger.info(f"Using pooling strategy: {encoder.pooling_strategy}")

    except Exception as e:
        logger.error(f"Error initializing {args.encoder_type.upper()} encoder: {e}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1
    
    # Initialize visualizer
    try:
        visualizer = HLAEmbeddingVisualizer(encoder)
    except Exception as e:
        logger.error(f"Error initializing visualizer: {e}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1
    
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
                pca_components=args.pca_components,
                debug=args.debug
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing locus {locus}: {e}")
            if args.debug:
                logger.error(traceback.format_exc())
            results.append({
                "locus": locus,
                "status": "error",
                "reason": str(e),
                "allele_count": 0
            })
    
    # Generate report
    try:
        report_file = write_report(results, reports_dir)
        logger.info(f"Analysis report written to {report_file}")
    except Exception as e:
        logger.error(f"Error writing report: {e}")
        if args.debug:
            logger.error(traceback.format_exc())
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"Analysis complete: {success_count} loci processed successfully, "
                f"{skipped_count} skipped, {error_count} errors")
    
    # Return success if at least one locus was processed successfully
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
