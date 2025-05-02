"""
Test HLAEncoder Base Class
-------------------------
Tests for the HLAEncoder base class.
"""
import os
import sys
import pytest
import pickle
import tempfile
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import HLAEncoder

# Create a concrete implementation of HLAEncoder for testing
class TestEncoder(HLAEncoder):
    """Concrete HLAEncoder implementation for testing"""
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Mock implementation of _encode_sequence"""
        # Simply return a vector with length equal to sequence length
        return np.ones(len(sequence))

class TestHLAEncoder:
    """Tests for HLAEncoder"""
    
    @pytest.fixture
    def sequence_file(self):
        """Create a temporary sequence file for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp:
            # Create a sample sequences dictionary
            sequences = {
                'A*01:01': 'MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDRNTRNVKAQSQTDRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLFGAVIAGAVVAAVMWRRKSSDRKGGSYSQAAVSHDSAQGSDVSLTACKV',
                'A*02:01': 'MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAAVSDPDSAQGSDVSLTACKV',
                'B*07:02': 'MLVMAPRTVLLLLSAALALTETWAGSHSMRYFYTAMSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRTEPRAPWVEQEGPEYWDRNTQIYKAQAQTDRESLRNLRGYYNQSEAGSHTLQRMYGCDLGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGLCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQSTIPIVGIVAGLAVLAVVVIGAVVATVMCRRKSSGGKGGSYSQAASSDSAQGSDVSLTACKV'
            }
            pickle.dump(sequences, temp)
            return temp.name
    
    @pytest.fixture
    def cache_dir(self):
        """Create a temporary cache directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            return temp_dir
    
    def test_initialization(self, sequence_file, cache_dir):
        """Test HLAEncoder initialization"""
        encoder = TestEncoder(sequence_file, cache_dir)
        assert len(encoder.sequences) == 3
        assert 'A*01:01' in encoder.sequences
        assert 'A*02:01' in encoder.sequences
        assert 'B*07:02' in encoder.sequences
    
    def test_get_embedding(self, sequence_file, cache_dir):
        """Test get_embedding method"""
        encoder = TestEncoder(sequence_file, cache_dir)
        
        # Get embedding for A*01:01
        embedding = encoder.get_embedding('A*01:01')
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (encoder.sequences['A*01:01'].__len__(),)
        
        # Check that embedding is cached
        assert 'A*01:01' in encoder.embeddings
        
        # Force regeneration
        embedding_2 = encoder.get_embedding('A*01:01', force=True)
        assert np.array_equal(embedding, embedding_2)
    
    def test_standardize_allele(self, sequence_file, cache_dir):
        """Test allele standardization"""
        encoder = TestEncoder(sequence_file, cache_dir, locus='A')
        
        # Test with HLA- prefix
        assert encoder._standardize_allele('HLA-A*01:01') == 'A*01:01'
        
        # Test with no * separator
        assert encoder._standardize_allele('A0101') == 'A*01:01'
        
        # Test with just digits
        assert encoder._standardize_allele('0101') == 'A*01:01'
    
    def test_batch_encode_alleles(self, sequence_file, cache_dir):
        """Test batch_encode_alleles method"""
        encoder = TestEncoder(sequence_file, cache_dir)
        
        # Batch encode two alleles
        alleles = ['A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles)
        
        assert len(results) == 2
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Force regeneration
        results_2 = encoder.batch_encode_alleles(alleles, force=True)
        assert len(results_2) == 2
        assert np.array_equal(results['A*01:01'], results_2['A*01:01'])
        assert np.array_equal(results['A*02:01'], results_2['A*02:01'])