"""
Comprehensive tests for IMGT data processing modules.
Tests IMGTDownloader and IMGTParser functionality.
"""
import pytest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.data.imgt_downloader import IMGTDownloader
from src.data.imgt_parser import IMGTParser


class TestIMGTDownloader:
    """Test suite for IMGTDownloader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = IMGTDownloader(data_dir=self.temp_dir)
    
    def test_initialization(self):
        """Test downloader initialization."""
        assert self.downloader.data_dir == Path(self.temp_dir)
        assert self.downloader.data_dir.exists()
        assert self.downloader.ftp_timeout == 30
        assert self.downloader.use_github_first is False
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        downloader = IMGTDownloader(
            data_dir=self.temp_dir,
            use_github_first=True,
            ftp_timeout=60
        )
        assert downloader.use_github_first is True
        assert downloader.ftp_timeout == 60
    
    def test_initialization_type_errors(self):
        """Test initialization raises TypeError for invalid inputs."""
        with pytest.raises(TypeError):
            IMGTDownloader(data_dir=123)
        
        with pytest.raises(TypeError):
            IMGTDownloader(use_github_first="yes")
        
        with pytest.raises(ValueError):
            IMGTDownloader(ftp_timeout=0)
        
        with pytest.raises(ValueError):
            IMGTDownloader(ftp_timeout=-1)
    
    def test_get_current_version_none(self):
        """Test get_current_version returns None if version file doesn't exist."""
        version = self.downloader._get_current_version()
        assert version is None
    
    def test_get_current_version_exists(self):
        """Test get_current_version returns version if file exists."""
        version_file = self.downloader.data_dir / "version.txt"
        version_file.write_text("3.49.0")
        
        version = self.downloader._get_current_version()
        assert version == "3.49.0"
    
    @patch('src.data.imgt_downloader.requests.get')
    def test_get_latest_version_github(self, mock_get):
        """Test getting latest version from GitHub."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'tag_name': 'v3.50.0'}
        mock_get.return_value = mock_response
        
        version = self.downloader._get_latest_version()
        assert version == 'v3.50.0'
    
    @patch('src.data.imgt_downloader.ftplib.FTP')
    @patch('src.data.imgt_downloader.requests.get')
    def test_download_latest_runtime_error(self, mock_get, mock_ftp):
        """Test download_latest raises RuntimeError when all sources fail."""
        mock_get.side_effect = Exception("Network error")
        mock_ftp.side_effect = Exception("FTP error")
        
        with pytest.raises(RuntimeError, match="Failed to download from all available sources"):
            self.downloader.download_latest()


class TestIMGTParser:
    """Test suite for IMGTParser class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.imgt_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "processed"
        self.imgt_dir.mkdir(parents=True)
        
        self.parser = IMGTParser(
            imgt_dir=str(self.imgt_dir),
            output_dir=str(self.output_dir)
        )
    
    def test_initialization(self):
        """Test parser initialization."""
        assert self.parser.imgt_dir == self.imgt_dir
        assert self.parser.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_initialization_type_errors(self):
        """Test initialization raises TypeError for invalid inputs."""
        with pytest.raises(TypeError):
            IMGTParser(imgt_dir=123)
        
        with pytest.raises(TypeError):
            IMGTParser(output_dir=None)
    
    def test_parse_protein_sequences_missing_directory(self):
        """Test parse_protein_sequences raises error if FASTA directory missing."""
        # Create parser but don't create FASTA files
        with pytest.raises(FileNotFoundError, match="FASTA directory not found"):
            self.parser.parse_protein_sequences()
    
    def test_parse_protein_sequences_with_fasta_dir(self):
        """Test parsing protein sequences from individual locus files."""
        # Create fasta directory with sample file
        fasta_dir = self.imgt_dir / "fasta"
        fasta_dir.mkdir()
        
        # Create a simple FASTA file
        test_fasta = fasta_dir / "A_prot.fasta"
        test_fasta.write_text(
            ">HLA:HLA00001 A*01:01:01:01 365 bp\n"
            "MAVMAPRTLLLILSGALALTETWAG\n"
            ">HLA:HLA00002 A*01:01:01:02N 365 bp\n"
            "MAVMAPRTLLLILSGALALTETWAG\n"
        )
        
        sequences = self.parser.parse_protein_sequences()
        
        # Check results
        assert len(sequences) > 0
        assert (self.output_dir / "hla_sequences.pkl").exists()
        assert (self.output_dir / "hla_sequences.txt").exists()
    
    def test_extract_allele_name(self):
        """Test allele name extraction from FASTA description."""
        # Test standard format
        description = "HLA:HLA00001 A*01:01:01:01 365 bp"
        allele = self.parser._extract_allele_name(description)
        assert allele == "A*01:01"
        
        # Test with different format
        description = "A*02:01:01"
        allele = self.parser._extract_allele_name(description)
        assert allele == "A*02:01"


class TestIMGTParserFallbackParser:
    """Test the fallback FASTA parser when BioPython not available."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.imgt_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "processed"
        self.imgt_dir.mkdir(parents=True)
        
        # Mock BioPython as unavailable
        import src.data.imgt_parser as parser_module
        self.original_seqio = parser_module.SeqIO
        parser_module.SeqIO = None
        
        self.parser = IMGTParser(
            imgt_dir=str(self.imgt_dir),
            output_dir=str(self.output_dir)
        )
    
    def teardown_method(self):
        """Restore BioPython."""
        import src.data.imgt_parser as parser_module
        parser_module.SeqIO = self.original_seqio
    
    def test_fallback_parser_used(self):
        """Test that fallback parser is used when BioPython unavailable."""
        # Create test FASTA file
        fasta_dir = self.imgt_dir / "fasta"
        fasta_dir.mkdir()
        
        test_fasta = fasta_dir / "A_prot.fasta"
        test_fasta.write_text(
            ">A*01:01\n"
            "MAVMAPRTL\n"
            ">A*02:01\n"
            "GSHSMRYFF\n"
        )
        
        # Parse using fallback
        sequences = self.parser._parse_fasta_fallback(test_fasta)
        
        assert len(sequences) == 2
        assert "A*01:01" in sequences
        assert "A*02:01" in sequences
        assert sequences["A*01:01"] == "MAVMAPRTL"
        assert sequences["A*02:01"] == "GSHSMRYFF"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
