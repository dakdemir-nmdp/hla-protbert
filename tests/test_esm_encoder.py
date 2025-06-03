"""
Test ESM Encoder
---------------
Tests for the ESMEncoder class.
"""
import os
import sys
import pytest
import pickle
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the transformers and torch imports for testing
class MockESM:
    def __init__(self, *args, **kwargs):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        # Mock tokenizer methods
        self.tokenizer.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 2
        
        # Mock model output
        self.model.return_value = MagicMock()
        self.model.return_value.last_hidden_state = MagicMock()


# Apply mocks before importing the module
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['transformers'].AutoTokenizer = MagicMock()
sys.modules['transformers'].AutoModel = MagicMock()
sys.modules['transformers'].logging = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['huggingface_hub'].login = MagicMock()

from src.models.encoders.esm import ESMEncoder

class TestESMEncoder:
    """Tests for ESMEncoder"""
    
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
    
    @patch.object(ESMEncoder, '_load_model')
    def test_initialization(self, mock_load_model, sequence_file, cache_dir):
        """Test ESMEncoder initialization"""
        mock_load_model.return_value = None
        
        # Initialize with minimal parameters
        encoder = ESMEncoder(sequence_file, cache_dir)
        
        assert encoder.model_name == "facebook/esm2_t33_650M_UR50D"
        assert encoder.pooling_strategy == "mean"
        assert encoder.device in ['cpu', 'cuda']
        
        # Initialize with custom parameters
        encoder = ESMEncoder(
            sequence_file, 
            cache_dir,
            model_name="facebook/esm2_t12_35M_UR50D",
            locus="A",
            device="cpu",
            pooling_strategy="cls"
        )
        
        assert encoder.model_name == "facebook/esm2_t12_35M_UR50D"
        assert encoder.pooling_strategy == "cls"
        assert encoder.device == "cpu"
        assert encoder.locus == "A"
        
    @patch.object(ESMEncoder, '_load_model')
    @patch.object(ESMEncoder, 'batch_encode')
    def test_batch_encode_alleles_with_force(self, mock_batch_encode, mock_load_model, 
                                            sequence_file, cache_dir):
        """Test batch_encode_alleles with force parameter"""
        mock_load_model.return_value = None
        # Mock batch_encode to return a numpy array with the correct shape
        mock_batch_encode.return_value = np.ones((1, 768))  # Return array for 1 sequence
        
        encoder = ESMEncoder(sequence_file, cache_dir, locus="A")
        
        # Add a mock embedding to the cache
        encoder.embeddings = {'A*01:01': np.ones(768)}
        
        # Test without force (should use cache for A*01:01)
        alleles = ['A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles, force=False)
        
        # Only A*02:01 should be encoded (A*01:01 is in cache)
        assert mock_batch_encode.call_count == 1
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Reset mock
        mock_batch_encode.reset_mock()
        # Update mock to return 2 embeddings for force=True test
        mock_batch_encode.return_value = np.ones((2, 768))
        
        # Test with force=True (should re-encode both)
        results = encoder.batch_encode_alleles(alleles, force=True)
        
        # Both alleles should be encoded
        assert mock_batch_encode.call_count == 1  # batch_encode is called once with both sequences
        assert 'A*01:01' in results
        assert 'A*02:01' in results