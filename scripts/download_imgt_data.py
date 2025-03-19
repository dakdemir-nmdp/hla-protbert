#!/usr/bin/env python3
"""
This script downloads HLA sequence data from the IMGT/HLA database.

Dependencies:
  - Project-specific src.data.imgt_downloader

Usage:
  python scripts/download_imgt_data.py --data-dir data/raw
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for module imports
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

# Import project modules
from src.data.imgt_downloader import IMGTDownloader
from src.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Download HLA sequences from IMGT/HLA database.")
    parser.add_argument('--data-dir', default='data/raw', 
                        help='Directory to store downloaded data')
    parser.add_argument('--force', action='store_true', 
                        help='Force download even if data exists')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Downloading HLA data to {data_dir}...")
    
    # Initialize downloader
    try:
        downloader = IMGTDownloader(data_dir=str(data_dir))
        
        # Check if data already exists
        if not args.force and (data_dir / "hla_prot.fasta").exists():
            logger.info("HLA data already exists. Use --force to download again.")
        else:
            # Download data
            logger.info("Downloading latest IMGT/HLA data...")
            downloader.download_latest(force=args.force)
            logger.info("Download complete!")
            
        # Print status
        if (data_dir / "hla_prot.fasta").exists():
            logger.info("HLA data is ready for encoding.")
        else:
            logger.error("Download failed: hla_prot.fasta not found.")
            return 1
            
    except Exception as e:
        logger.error(f"Error downloading HLA data: {e}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())
