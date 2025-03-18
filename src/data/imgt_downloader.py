"""
IMGT/HLA Database Downloader
----------------------------
Tools to download and update the IMGT/HLA database containing HLA allele sequences.
"""
import os
import logging
import ftplib
import zipfile
import requests
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class IMGTDownloader:
    """Handles downloading and updating IMGT/HLA database files"""
    
    IMGT_FTP_SERVER = "ftp.ebi.ac.uk"
    IMGT_FTP_DIR = "/pub/databases/ipd/imgt/hla/fasta/"
    IMGT_VERSION_URL = "https://www.ebi.ac.uk/ipd/imgt/hla/release.html"
    
    def __init__(self, data_dir='./data/raw'):
        """Initialize downloader with target directory
        
        Args:
            data_dir: Directory to store downloaded IMGT/HLA data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def download_latest(self, force=False):
        """Download the latest version of IMGT/HLA database
        
        Args:
            force: If True, download even if version is current
        
        Returns:
            bool: True if updated, False if already current
        """
        # Check current version
        current_version = self._get_current_version()
        
        # Get latest version from IMGT/HLA website
        latest_version = self._get_latest_version()
        
        if force or current_version != latest_version:
            logger.info(f"Updating IMGT/HLA from {current_version} to {latest_version}")
            # Download FTP data
            self._download_ftp_data()
            # Extract files
            self._extract_data()
            # Update version info
            self._update_version_info(latest_version)
            return True
        else:
            logger.info(f"IMGT/HLA database already at latest version {current_version}")
            return False
            
    def _get_current_version(self):
        """Get currently installed version
        
        Returns:
            str or None: Current version or None if not installed
        """
        version_file = self.data_dir / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return None
    
    def _get_latest_version(self):
        """Get latest version available from IMGT/HLA website
        
        Returns:
            str: Latest version string (e.g., "3.49.0")
        """
        try:
            response = requests.get(self.IMGT_VERSION_URL)
            response.raise_for_status()
            
            # Extract version from page content
            # This is a simplified approach - might need adjustment based on actual page structure
            content = response.text
            if "Version" in content:
                idx = content.find("Version")
                version_str = content[idx:idx+20]
                version = version_str.split()[1].strip()
                return version
            
            # Fallback to date-based versioning if version not found
            today = datetime.now().strftime("%Y%m%d")
            return f"unknown-{today}"
            
        except Exception as e:
            logger.error(f"Error getting latest IMGT/HLA version: {e}")
            # Fallback to date-based versioning
            today = datetime.now().strftime("%Y%m%d")
            return f"error-{today}"
    
    def _download_ftp_data(self):
        """Download data files from IMGT/HLA FTP server"""
        target_dir = self.data_dir / "fasta"
        target_dir.mkdir(exist_ok=True)
        
        try:
            # Connect to FTP server
            ftp = ftplib.FTP(self.IMGT_FTP_SERVER)
            ftp.login()
            ftp.cwd(self.IMGT_FTP_DIR)
            
            # Get list of files
            file_list = []
            ftp.retrlines('LIST', lambda x: file_list.append(x.split()[-1]))
            
            # Download each FASTA file
            for filename in file_list:
                if filename.endswith('_prot.fasta'):
                    logger.info(f"Downloading {filename}")
                    with open(target_dir / filename, 'wb') as fp:
                        ftp.retrbinary(f'RETR {filename}', fp.write)
            
            # Download consolidated files if available
            try:
                ftp.cwd('..')  # Move up to main directory
                with open(self.data_dir / "hla_prot.fasta", 'wb') as fp:
                    ftp.retrbinary('RETR hla_prot.fasta', fp.write)
            except:
                logger.warning("Consolidated hla_prot.fasta not found, using individual files")
                
            ftp.quit()
            
        except Exception as e:
            logger.error(f"Error downloading IMGT/HLA data: {e}")
            raise
    
    def _extract_data(self):
        """Extract any zip files if present"""
        for zip_file in self.data_dir.glob("*.zip"):
            logger.info(f"Extracting {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
    
    def _update_version_info(self, version):
        """Update the stored version information
        
        Args:
            version: Version string to save
        """
        with open(self.data_dir / "version.txt", 'w') as f:
            f.write(version)
        
        # Also save download date
        with open(self.data_dir / "download_date.txt", 'w') as f:
            f.write(datetime.now().isoformat())
