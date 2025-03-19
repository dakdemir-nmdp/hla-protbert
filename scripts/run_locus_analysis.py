#!/usr/bin/env python
"""
Run HLA Locus Embedding Analysis
-------------------------------
Example script to run the locus-specific embedding analysis with custom parameters.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the locus analysis with custom parameters"""
    # Define common loci
    class1_loci = ["A", "B", "C"]
    class2_loci = ["DRB1", "DQB1", "DPB1"]
    
    # Output directory
    output_dir = "./data/analysis/locus_embeddings"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis for Class I loci
    print("Running analysis for Class I HLA loci...")
    subprocess.run([
        "python", "scripts/analyze_locus_embeddings.py",
        "--loci", *class1_loci,
        "--output-dir", f"{output_dir}/class1",
        "--umap-neighbors", "15",
        "--umap-min-dist", "0.1",
        "--tsne-perplexity", "30",
        "--tsne-learning-rate", "200",
        "--batch-size", "8",
        "--verbose"
    ])
    
    # Run analysis for Class II loci
    print("\nRunning analysis for Class II HLA loci...")
    subprocess.run([
        "python", "scripts/analyze_locus_embeddings.py",
        "--loci", *class2_loci,
        "--output-dir", f"{output_dir}/class2",
        "--umap-neighbors", "20",  # Different parameters for Class II
        "--umap-min-dist", "0.2",
        "--tsne-perplexity", "40",
        "--tsne-learning-rate", "150",
        "--batch-size", "8",
        "--verbose"
    ])
    
    print("\nAnalysis complete. Results saved to:")
    print(f"- Class I: {output_dir}/class1")
    print(f"- Class II: {output_dir}/class2")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
