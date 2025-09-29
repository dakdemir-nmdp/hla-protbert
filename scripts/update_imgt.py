#!/usr/bin/env python
"""
Update IMGT/HLA Database
-----------------------
Script to download and update the IMGT/HLA database.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import modules
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from src.data.imgt_downloader import IMGTDownloader
from src.data.imgt_parser import IMGTParser
from src.utils.logging import setup_logging
from src.utils.config import ConfigManager

def main():
    """Main function to update IMGT/HLA database"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update IMGT/HLA database")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory")
    parser.add_argument("--raw-dir", dest="raw_dir", help="Directory for raw IMGT/HLA data")
    parser.add_argument("--processed-dir", dest="processed_dir", help="Directory for processed data")
    parser.add_argument("--force", action="store_true", help="Force update even if already current")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Determine directories
    data_dir = args.data_dir or config.get("data.base_dir", str(project_dir / "data"))
    raw_dir = args.raw_dir or config.get("data.raw_dir", os.path.join(data_dir, "raw"))
    processed_dir = args.processed_dir or config.get("data.processed_dir", os.path.join(data_dir, "processed"))
    
    # Create downloader
    logger.info(f"Initializing IMGT/HLA downloader (raw_dir={raw_dir})")
    downloader = IMGTDownloader(data_dir=raw_dir)
    
    # Download latest version
    logger.info("Checking for IMGT/HLA database updates...")
    updated = downloader.download_latest(force=args.force)
    
    if not updated and not args.force:
        logger.info("IMGT/HLA database is already up to date. Use --force to update anyway.")
        return
    
    # Parse downloaded data
    logger.info(f"Parsing IMGT/HLA data (output_dir={processed_dir})")
    parser = IMGTParser(imgt_dir=raw_dir, output_dir=processed_dir)
    sequences = parser.parse_protein_sequences()
    
    logger.info(f"IMGT/HLA database update complete. Parsed {len(sequences)} sequences.")

if __name__ == "__main__":
    main()
