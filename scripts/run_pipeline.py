#!/usr/bin/env python3
"""
HLA-ProtBERT Pipeline
-------------------
This script coordinates the full HLA-ProtBERT pipeline by:
1. Downloading HLA protein data from IMGT/HLA database
2. Encoding sequences using ProtBERT
3. (Optionally) Generating visualizations

Usage:
  python scripts/run_pipeline.py --data-dir data/raw --output-dir data/processed
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Add parent directory to path for module imports
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.utils.logging import setup_logging

def run_command(cmd, description=None):
    """Run a command and return the exit code"""
    if description:
        logger.info(f"{description}...")
    
    logger.debug(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        logger.error(f"STDOUT: {result.stdout}")
    else:
        logger.info(f"Command completed successfully")
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.debug(f"STDOUT: {line}")
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run the complete HLA-ProtBERT pipeline.")
    parser.add_argument('--data-dir', default='data/raw', 
                        help='Directory to store downloaded data')
    parser.add_argument('--output-dir', default='data/processed', 
                        help='Directory to save outputs')
    parser.add_argument('--force', action='store_true', 
                        help='Force download even if data exists')
    parser.add_argument('--locus', 
                        help='Specific HLA locus to process (e.g., A, B, C)')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download data, skip encoding')
    parser.add_argument('--encode-only', action='store_true',
                        help='Skip download, only perform encoding')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (extra verbosity)')
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug or args.verbose else "INFO"
    global logger
    logger = setup_logging(level=log_level)
    
    # Create relevant directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare common arguments
    verbose_arg = ['-v'] if args.verbose or args.debug else []
    
    exit_code = 0
    
    # Step 1: Download data
    if not args.encode_only:
        download_cmd = [
            sys.executable,
            str(script_dir / 'download_imgt_data.py'),
            f'--data-dir={args.data_dir}'
        ]
        
        if args.force:
            download_cmd.append('--force')
            
        download_cmd.extend(verbose_arg)
        
        logger.info("=== STEP 1: DOWNLOADING HLA DATA ===")
        exit_code = run_command(download_cmd, "Downloading HLA data")
        
        if exit_code != 0:
            logger.error("Data download failed, aborting pipeline")
            return exit_code
    
    # Step 2: Encode sequences
    if not args.download_only and exit_code == 0:
        encode_cmd = [
            sys.executable,
            str(script_dir / 'encode_sequences.py'),
            f'--data-dir={args.data_dir}',
            f'--output-dir={args.output_dir}'
        ]
        
        if args.locus:
            encode_cmd.append(f'--locus={args.locus}')
            
        if args.skip_visualizations:
            encode_cmd.append('--skip-visualizations')
            
        encode_cmd.extend(verbose_arg)
        
        logger.info("=== STEP 2: ENCODING SEQUENCES ===")
        exit_code = run_command(encode_cmd, "Encoding HLA sequences")
        
        if exit_code != 0:
            logger.error("Sequence encoding failed")
            return exit_code
    
    # Done
    if exit_code == 0:
        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        
        # Print output locations
        if not args.encode_only:
            logger.info(f"Downloaded data: {data_dir.absolute()}")
        if not args.download_only:
            logger.info(f"Processed data: {output_dir.absolute()}")
            logger.info(f"Embeddings: {output_dir.parent / 'embeddings'}")
            if not args.skip_visualizations:
                logger.info(f"Visualizations: {output_dir / 'plots'}")
    else:
        logger.error("=== PIPELINE FAILED ===")
    
    return exit_code

if __name__ == '__main__':
    exit(main())
