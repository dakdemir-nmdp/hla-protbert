"""
Test ProtBERT Encoder
--------------------
Tests for the ProtBERTEncoder class.
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
class MockBERT:
    def __init__(self, *args, **kwargs):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        # Mock tokenizer methods
        self.tokenizer.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        
        # Mock model output
        self.model.return_value = MagicMock()
        self.model.return_value.last_hidden_state = MagicMock()


# Apply mocks before importing the module
mock_torch = MagicMock()
# Create a proper Dataset base class mock
class MockDataset:
    def __init__(self):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        return {}

mock_torch.utils.data.Dataset = MockDataset
sys.modules['torch'] = mock_torch
sys.modules['transformers'] = MagicMock()
sys.modules['transformers'].BertModel = MagicMock()
sys.modules['transformers'].BertTokenizer = MagicMock()
sys.modules['transformers'].BertForSequenceClassification = MagicMock()
sys.modules['transformers'].TrainingArguments = MagicMock()
sys.modules['transformers'].Trainer = MagicMock()

# Import the module after mocking
from src.models.encoders.protbert import ProtBERTEncoder, HLADataset

class TestProtBERTEncoder:
    """Tests for ProtBERTEncoder"""
    
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
    
    @patch.object(ProtBERTEncoder, '_load_model')
    def test_initialization(self, mock_load_model, sequence_file, cache_dir):
        """Test ProtBERTEncoder initialization"""
        mock_load_model.return_value = None
        
        # Initialize with minimal parameters
        encoder = ProtBERTEncoder(sequence_file, cache_dir)
        
        assert encoder.model_name == "Rostlab/prot_bert"
        assert encoder.pooling_strategy == "mean"
        assert encoder.device in ['cpu', 'cuda']
        assert encoder.use_peptide_binding_region is True
        
        # Initialize with custom parameters
        encoder = ProtBERTEncoder(
            sequence_file, 
            cache_dir,
            model_name="Rostlab/prot_bert_bfd",
            locus="A",
            device="cpu",
            pooling_strategy="cls",
            use_peptide_binding_region=False
        )
        
        assert encoder.model_name == "Rostlab/prot_bert_bfd"
        assert encoder.pooling_strategy == "cls"
        assert encoder.device == "cpu"
        assert encoder.locus == "A"
        assert encoder.use_peptide_binding_region is False
    
    @patch.object(ProtBERTEncoder, '_load_model')
    @patch.object(ProtBERTEncoder, 'batch_encode')
    def test_batch_encode_alleles_with_force(self, mock_batch_encode, mock_load_model, 
                                           sequence_file, cache_dir):
        """Test batch_encode_alleles with force parameter"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = np.ones((2, 768))  # Mock embedding vectors for batch
        
        encoder = ProtBERTEncoder(sequence_file, cache_dir, locus="A")
        
        # Add a mock embedding to the cache
        encoder.embeddings = {'A*01:01': np.ones(768)}
        
        # Test without force (should use cache for A*01:01)
        alleles = ['A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles, force=False)
        
        # Only A*02:01 should be encoded (A*01:01 is in cache)
        sequences_encoded = mock_batch_encode.call_args[0][0]
        assert len(sequences_encoded) == 1  # Only one sequence should be encoded
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Reset mock
        mock_batch_encode.reset_mock()
        
        # Test with force=True (should re-encode both)
        results = encoder.batch_encode_alleles(alleles, force=True)
        
        # Both alleles should be encoded
        sequences_encoded = mock_batch_encode.call_args[0][0]
        assert len(sequences_encoded) == 2  # Both sequences should be encoded
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
