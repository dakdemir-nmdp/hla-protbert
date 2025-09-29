"""
IMGT/HLA Database Downloader
----------------------------
Tools to download and update the IMGT/HLA database containing HLA allele sequences.
Includes fallback to GitHub repository for cloud environments.
"""
import os
import logging
import ftplib
import zipfile
import requests
from pathlib import Path
from datetime import datetime
import json
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class IMGTDownloader:
    """Handles downloading and updating IMGT/HLA database files"""
    
    # FTP settings (primary source)
    IMGT_FTP_SERVER = "ftp.ebi.ac.uk"
    IMGT_FTP_DIR = "/pub/databases/ipd/imgt/hla/fasta/"
    IMGT_VERSION_URL = "https://www.ebi.ac.uk/ipd/imgt/hla/docs/release.html"
    
    # GitHub settings (fallback source)
    GITHUB_REPO = "ANHIG/IMGTHLA"
    GITHUB_API_URL = "https://api.github.com/repos/ANHIG/IMGTHLA"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/ANHIG/IMGTHLA"
    
    def __init__(self, data_dir='./data/raw', use_github_first=False, ftp_timeout=30):
        """Initialize downloader with target directory
        
        Args:
            data_dir: Directory to store downloaded IMGT/HLA data
            use_github_first: If True, try GitHub before FTP
            ftp_timeout: Timeout for FTP connections in seconds
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.use_github_first = use_github_first
        self.ftp_timeout = ftp_timeout
        
    def download_latest(self, force=False):
        """Download the latest version of IMGT/HLA database
        
        Args:
            force: If True, download even if version is current
        
        Returns:
            bool: True if updated, False if already current
        """
        success = False
        
        # Try different download methods
        download_methods = [
            ("GitHub", self._download_github_data),
            ("FTP", self._download_ftp_data)
        ] if self.use_github_first else [
            ("FTP", self._download_ftp_data),
            ("GitHub", self._download_github_data)
        ]
        
        for method_name, method in download_methods:
            try:
                logger.info(f"Attempting download from {method_name}...")
                method()
                success = True
                logger.info(f"Successfully downloaded from {method_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to download from {method_name}: {e}")
                continue
        
        if not success:
            raise RuntimeError("Failed to download from all available sources (FTP and GitHub)")
        
        # Extract files
        self._extract_data()
        
        # Update version info with current date
        today = datetime.now().strftime("%Y%m%d")
        self._update_version_info(f"download-{today}")
        return True
            
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
        """Get latest version available from IMGT/HLA website or GitHub
        
        Returns:
            str: Latest version string (e.g., "3.49.0")
        """
        # Try GitHub first for version info
        try:
            response = requests.get(f"{self.GITHUB_API_URL}/releases/latest", timeout=10)
            if response.status_code == 200:
                release_data = response.json()
                return release_data['tag_name']
        except Exception as e:
            logger.debug(f"Could not get version from GitHub releases: {e}")
        
        # Try GitHub tags
        try:
            response = requests.get(f"{self.GITHUB_API_URL}/tags", timeout=10)
            if response.status_code == 200:
                tags = response.json()
                if tags:
                    return tags[0]['name']
        except Exception as e:
            logger.debug(f"Could not get version from GitHub tags: {e}")
        
        # Fallback to original method
        try:
            response = requests.get(self.IMGT_VERSION_URL, timeout=10)
            response.raise_for_status()
            
            # Extract version from page content
            content = response.text
            if "Version" in content:
                idx = content.find("Version")
                version_str = content[idx:idx+20]
                version = version_str.split()[1].strip()
                return version
            
        except Exception as e:
            logger.error(f"Error getting latest IMGT/HLA version: {e}")
        
        # Final fallback to date-based versioning
        today = datetime.now().strftime("%Y%m%d")
        return f"unknown-{today}"
    
    def _download_github_data(self):
        """Download data files from GitHub repository"""
        target_dir = self.data_dir / "fasta"
        target_dir.mkdir(exist_ok=True)
        
        # Get the default branch
        try:
            response = requests.get(f"{self.GITHUB_API_URL}", timeout=10)
            response.raise_for_status()
            repo_info = response.json()
            default_branch = repo_info['default_branch']
        except Exception as e:
            logger.warning(f"Could not determine default branch, using 'Latest': {e}")
            default_branch = "Latest"
        
        # Get directory contents to find FASTA files
        fasta_dirs = ['fasta', 'alignments']  # Common directories that might contain FASTA files
        downloaded_files = 0
        
        for fasta_dir in fasta_dirs:
            try:
                # Get contents of the fasta directory
                response = requests.get(
                    f"{self.GITHUB_API_URL}/contents/{fasta_dir}",
                    params={'ref': default_branch},
                    timeout=30
                )
                
                if response.status_code != 200:
                    continue
                    
                contents = response.json()
                
                # Download protein FASTA files
                for item in contents:
                    if item['type'] == 'file' and item['name'].endswith('_prot.fasta'):
                        file_url = item['download_url']
                        filename = item['name']
                        
                        logger.info(f"Downloading {filename} from GitHub")
                        file_response = requests.get(file_url, timeout=60)
                        file_response.raise_for_status()
                        
                        with open(target_dir / filename, 'wb') as fp:
                            fp.write(file_response.content)
                        
                        downloaded_files += 1
                        
            except Exception as e:
                logger.debug(f"Could not access {fasta_dir} directory: {e}")
                continue
        
        # Try to download consolidated file
        try:
            consolidated_url = f"{self.GITHUB_RAW_URL}/{default_branch}/hla_prot.fasta"
            response = requests.get(consolidated_url, timeout=60)
            if response.status_code == 200:
                logger.info("Downloading consolidated hla_prot.fasta from GitHub")
                with open(self.data_dir / "hla_prot.fasta", 'wb') as fp:
                    fp.write(response.content)
                downloaded_files += 1
        except Exception as e:
            logger.debug(f"Could not download consolidated file: {e}")
        
        if downloaded_files == 0:
            raise RuntimeError("No FASTA files found or downloaded from GitHub")
        
        logger.info(f"Downloaded {downloaded_files} files from GitHub")
    
    def _download_ftp_data(self):
        """Download data files from IMGT/HLA FTP server"""
        target_dir = self.data_dir / "fasta"
        target_dir.mkdir(exist_ok=True)
        
        try:
            # Connect to FTP server with timeout
            ftp = ftplib.FTP(self.IMGT_FTP_SERVER, timeout=self.ftp_timeout)
            ftp.login()
            ftp.cwd(self.IMGT_FTP_DIR)
            
            # Get list of files
            file_list = []
            ftp.retrlines('LIST', lambda x: file_list.append(x.split()[-1]))
            
            # Download each FASTA file
            downloaded_files = 0
            for filename in file_list:
                if filename.endswith('_prot.fasta'):
                    logger.info(f"Downloading {filename}")
                    with open(target_dir / filename, 'wb') as fp:
                        ftp.retrbinary(f'RETR {filename}', fp.write)
                    downloaded_files += 1
            
            # Download consolidated files if available
            try:
                ftp.cwd('..')  # Move up to main directory
                with open(self.data_dir / "hla_prot.fasta", 'wb') as fp:
                    ftp.retrbinary('RETR hla_prot.fasta', fp.write)
                downloaded_files += 1
            except:
                logger.warning("Consolidated hla_prot.fasta not found, using individual files")
                
            ftp.quit()
            
            if downloaded_files == 0:
                raise RuntimeError("No files downloaded from FTP")
                
        except Exception as e:
            logger.error(f"Error downloading IMGT/HLA data via FTP: {e}")
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

# Convenience function for cloud environments
def download_imgt_data(data_dir='./data/raw', use_github_first=True):
    """
    Convenience function optimized for cloud environments like AWS SageMaker
    
    Args:
        data_dir: Directory to store data
        use_github_first: Use GitHub as primary source (recommended for cloud)
    
    Returns:
        bool: True if successful
    """
    downloader = IMGTDownloader(data_dir=data_dir, use_github_first=use_github_first)
    return downloader.download_latest()