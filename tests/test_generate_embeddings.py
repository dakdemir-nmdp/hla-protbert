"""
Test generate_embeddings.py
-------------------------
Tests for the generate_embeddings.py script functions.
"""
import os
import sys
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the relevant modules
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from scripts.generate_embeddings import load_alleles_from_file

class TestGenerateEmbeddings:
    """Tests for generate_embeddings.py functions"""
    
    def test_load_alleles_from_csv(self):
        """Test loading alleles from CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            # Create a sample CSV file
            temp.write("allele,sequence\n")
            temp.write("A*01:01,SEQUENCE1\n")
            temp.write("A*02:01,SEQUENCE2\n")
            temp.flush()
            
            # Test with default column name
            alleles = load_alleles_from_file(Path(temp.name))
            assert len(alleles) == 2
            assert 'A*01:01' in alleles
            assert 'A*02:01' in alleles
    
    def test_load_alleles_from_txt(self):
        """Test loading alleles from text file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
            # Create a sample text file
            temp.write("A*01:01\n")
            temp.write("A*02:01\n")
            temp.write("B*07:02\n")
            temp.flush()
            
            alleles = load_alleles_from_file(Path(temp.name))
            assert len(alleles) == 3
            assert 'A*01:01' in alleles
            assert 'A*02:01' in alleles
            assert 'B*07:02' in alleles
    
    def test_load_alleles_from_tsv(self):
        """Test loading alleles from TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as temp:
            # Create a sample TSV file
            temp.write("hla_allele\tsequence\n")
            temp.write("A*01:01\tSEQUENCE1\n")
            temp.write("A*02:01\tSEQUENCE2\n")
            temp.flush()
            
            # Test with custom column name
            alleles = load_alleles_from_file(Path(temp.name))
            assert len(alleles) == 2
            assert 'A*01:01' in alleles
            assert 'A*02:01' in alleles
    
    def test_load_alleles_unsupported_format(self):
        """Test loading alleles from unsupported file format"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp:
            temp.write('{"alleles": ["A*01:01", "A*02:01"]}')
            temp.flush()
            
            # Should return empty list for unsupported format
            alleles = load_alleles_from_file(Path(temp.name))
            assert len(alleles) == 0
    
    def test_load_alleles_file_not_found(self):
        """Test loading alleles from non-existent file"""
        # Should return empty list for non-existent file
        alleles = load_alleles_from_file(Path("/nonexistent/file.txt"))
        assert len(alleles) == 0