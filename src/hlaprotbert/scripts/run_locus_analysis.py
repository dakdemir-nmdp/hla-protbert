#!/usr/bin/env python
"""
Run HLA Locus Embedding Analysis
-------------------------------
Example script to run the locus-specific embedding analysis with custom parameters.
"""
import os
import subprocess
import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run HLA locus embedding analysis for Class I and Class II loci"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./data/analysis/locus_embeddings",
        help="Base directory for analysis outputs"
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
        "--class1-only",
        action="store_true",
        help="Only analyze Class I loci"
    )
    
    parser.add_argument(
        "--class2-only",
        action="store_true",
        help="Only analyze Class II loci"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more detailed error messages"
    )
    
    return parser.parse_args()

def setup_logging(debug=False, log_file=None):
    """Set up logging
    
    Args:
        debug: Enable debug mode (more verbose logging)
        log_file: Path to log file (None for console only)
        
    Returns:
        Logger instance
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Error setting up log file {log_file}: {e}")
    
    # Configure warnings
    if not debug:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
    
    return logging.getLogger("run_locus_analysis")

def run_analysis(cmd, logger, timeout=None):
    """Run analysis command and log results
    
    Args:
        cmd: Command to run as list of strings
        logger: Logger instance
        timeout: Timeout in seconds (None for no timeout)
        
    Returns:
        Boolean indicating success or failure
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout
        )
        
        # Check if command was successful
        if result.returncode == 0:
            logger.info("Command completed successfully")
        else:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"Error output:\n{result.stderr}")
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_input_files(args, logger):
    """Check that input files exist and are accessible
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Boolean indicating whether checks passed
    """
    # Check sequence file
    sequence_file = Path(args.sequence_file).resolve()
    if not sequence_file.exists():
        logger.error(f"Sequence file not found: {sequence_file}")
        return False
    logger.debug(f"Found sequence file: {sequence_file}")
    
    # Check cache directory exists or can be created
    cache_dir = Path(args.cache_dir).resolve()
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(exist_ok=True, parents=True)
            logger.debug(f"Created cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}")
            return False
    
    return True

def main():
    """Run the locus analysis with custom parameters"""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base output directory
    base_output_dir = Path(args.output_dir).resolve()
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log directory and file
    log_dir = base_output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"analysis_{timestamp}.log"
    
    # Setup logging with file
    logger = setup_logging(args.debug, log_file)
    logger.info(f"Started HLA locus analysis run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define loci
    class1_loci = ["A", "B", "C"]
    class2_loci = ["DRB1", "DQB1", "DPB1"]
    
    # Validate input paths
    if not check_input_files(args, logger):
        logger.error("Input validation failed. Please fix the issues above and try again.")
        return 1
    
    logger.debug(f"Logs will be saved to: {log_file}")
    
    # Determine which analyses to run
    run_class1 = not args.class2_only
    run_class2 = not args.class1_only
    
    # Track success/failure
    success_count = 0
    total_analyses = sum([run_class1, run_class2])
    
    # Run analysis for Class I loci
    if run_class1:
        logger.info("Running analysis for Class I HLA loci...")
        class1_output_dir = base_output_dir / "class1"
        
        cmd = [
            "python", "scripts/analyze_locus_embeddings.py",
            "--loci", *class1_loci,
            "--output-dir", str(class1_output_dir),
            "--sequence-file", args.sequence_file,
            "--cache-dir", args.cache_dir,
            "--umap-neighbors", "15",
            "--umap-min-dist", "0.1",
            "--tsne-perplexity", "30",
            "--tsne-learning-rate", "200",
            "--batch-size", "8",
            "--verbose"
        ]
        
        # Add debug flag if enabled
        if args.debug:
            cmd.append("--debug")
        
        # Run command and track success
        if run_analysis(cmd, logger):
            success_count += 1
            logger.info("Class I analysis completed successfully")
        else:
            logger.error("Class I analysis failed")
    
    # Run analysis for Class II loci
    if run_class2:
        logger.info("\nRunning analysis for Class II HLA loci...")
        class2_output_dir = base_output_dir / "class2"
        
        cmd = [
            "python", "scripts/analyze_locus_embeddings.py",
            "--loci", *class2_loci,
            "--output-dir", str(class2_output_dir),
            "--sequence-file", args.sequence_file,
            "--cache-dir", args.cache_dir,
            "--umap-neighbors", "20",  # Different parameters for Class II
            "--umap-min-dist", "0.2",
            "--tsne-perplexity", "40",
            "--tsne-learning-rate", "150",
            "--batch-size", "8",
            "--verbose"
        ]
        
        # Add debug flag if enabled
        if args.debug:
            cmd.append("--debug")
        
        # Run command and track success
        if run_analysis(cmd, logger):
            success_count += 1
            logger.info("Class II analysis completed successfully")
        else:
            logger.error("Class II analysis failed")
    
    # Print summary
    logger.info("\nAnalysis summary:")
    logger.info(f"- Analyses requested: {total_analyses}")
    logger.info(f"- Successful analyses: {success_count}")
    logger.info(f"- Failed analyses: {total_analyses - success_count}")
    
    if success_count > 0:
        logger.info("\nResults saved to:")
        if run_class1 and success_count > 0:
            logger.info(f"- Class I: {base_output_dir}/class1")
        if run_class2 and success_count > 0:
            logger.info(f"- Class II: {base_output_dir}/class2")
            
        # Generate combined report links
        class1_report = base_output_dir / "class1" / "reports" / "locus_analysis_report.md"
        class2_report = base_output_dir / "class2" / "reports" / "locus_analysis_report.md"
        
        if class1_report.exists():
            logger.info(f"- Class I report: {class1_report}")
        if class2_report.exists():
            logger.info(f"- Class II report: {class2_report}")
            
        logger.info("\nTo explore the results interactively, open the notebook:")
        logger.info("notebooks/locus_embeddings_analysis.ipynb")
    
    # Return success if at least one analysis completed successfully
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
