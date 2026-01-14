"""
Test ProtT5 Encoder
------------------
Tests for the ProtT5Encoder class.
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

from hlaprotbert.models.encoders.prott5 import ProtT5Encoder


class TestProtT5Encoder:
    """Tests for ProtT5Encoder"""
    
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
            yield temp_dir
    
    @patch.object(ProtT5Encoder, '_load_model')
    def test_initialization(self, mock_load_model, sequence_file, cache_dir):
        """Test ProtT5Encoder initialization with various configurations"""
        mock_load_model.return_value = None
        
        # Test default initialization
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        assert encoder.model_name == "Rostlab/prot_t5_xl_uniref50"
        assert encoder.pooling_strategy == "mean"
        assert encoder.device in ['cpu', 'cuda']
        assert str(encoder.cache_dir).endswith('prott5')
        
        # Test custom parameters
        encoder = ProtT5Encoder(
            sequence_file, 
            cache_dir,
            model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
            locus="A",
            device="cpu",
            pooling_strategy="last"
        )
        assert encoder.model_name == "Rostlab/prot_t5_xl_half_uniref50-enc"
        assert encoder.pooling_strategy == "last"
        assert encoder.device == "cpu"
        assert encoder.locus == "A"
        
    @patch.object(ProtT5Encoder, '_load_model')
    def test_invalid_pooling_strategy(self, mock_load_model, sequence_file, cache_dir):
        """Test that invalid pooling strategy raises ValueError"""
        mock_load_model.return_value = None
        
        with pytest.raises(ValueError, match="pooling_strategy must be 'mean' or 'last'"):
            ProtT5Encoder(
                sequence_file,
                cache_dir,
                pooling_strategy="invalid"
            )
    
    @patch.object(ProtT5Encoder, '_load_model')
    def test_cache_dir_structure(self, mock_load_model, sequence_file, cache_dir):
        """Test that cache directory is properly organized"""
        mock_load_model.return_value = None
        
        # Test that 'prott5' is added to cache path if not present
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        assert encoder.cache_dir.name == 'prott5'
        
        # Test that 'prott5' is not duplicated if already present
        prott5_cache = Path(cache_dir) / 'prott5'
        encoder = ProtT5Encoder(sequence_file, prott5_cache)
        assert encoder.cache_dir.name == 'prott5'
        assert str(encoder.cache_dir).count('prott5') == 1
    
    @patch.object(ProtT5Encoder, '_load_model')
    @patch.object(ProtT5Encoder, '_encode_sequence')
    def test_get_embedding(self, mock_encode, mock_load_model, sequence_file, cache_dir):
        """Test get_embedding method"""
        mock_load_model.return_value = None
        mock_encode.return_value = np.random.rand(1024)  # ProtT5 embedding dimension
        
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        
        # Get embedding for A*01:01
        embedding = encoder.get_embedding('A*01:01')
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        
        # Check that embedding is cached
        assert 'A*01:01' in encoder.embeddings
        
        # Test force regeneration
        embedding_2 = encoder.get_embedding('A*01:01', force=True)
        assert isinstance(embedding_2, np.ndarray)
        assert embedding_2.shape == (1024,)
    
    @patch.object(ProtT5Encoder, '_load_model')
    @patch.object(ProtT5Encoder, '_batch_encode_sequences')
    def test_batch_encode_alleles(self, mock_batch_encode, mock_load_model, 
                                  sequence_file, cache_dir):
        """Test batch_encode_alleles method"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = [
            np.random.rand(1024),
            np.random.rand(1024),
            np.random.rand(1024)
        ]
        
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        
        # Batch encode multiple alleles
        alleles = ['A*01:01', 'A*02:01', 'B*07:02']
        results = encoder.batch_encode_alleles(alleles, batch_size=4)
        
        assert len(results) == 3
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        assert 'B*07:02' in results
        assert all(isinstance(v, np.ndarray) for v in results.values())
        assert all(v.shape == (1024,) for v in results.values())
    
    @patch.object(ProtT5Encoder, '_load_model')
    @patch.object(ProtT5Encoder, '_batch_encode_sequences')
    def test_batch_encode_with_duplicates(self, mock_batch_encode, mock_load_model,
                                         sequence_file, cache_dir):
        """Test that batch encoding handles duplicates correctly"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = [
            np.random.rand(1024),
            np.random.rand(1024)
        ]
        
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        
        # Include duplicate alleles
        alleles = ['A*01:01', 'A*02:01', 'A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles, batch_size=4)
        
        # Should only return unique alleles (method deduplicates)
        assert len(results) == 2
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Should only call batch encode for unique ones
        mock_batch_encode.assert_called_once()
    
    @patch.object(ProtT5Encoder, '_load_model')
    @patch.object(ProtT5Encoder, '_batch_encode_sequences')
    def test_batch_encode_with_cached(self, mock_batch_encode, mock_load_model,
                                     sequence_file, cache_dir):
        """Test that batch encoding skips cached alleles"""
        mock_load_model.return_value = None
        
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        
        # Pre-cache one embedding
        encoder.embeddings['A*01:01'] = np.random.rand(1024)
        
        # Mock should only be called for non-cached allele
        mock_batch_encode.return_value = [np.random.rand(1024)]
        
        alleles = ['A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles, force=False)
        
        assert len(results) == 2
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Should only encode the non-cached allele
        mock_batch_encode.assert_called_once()
    
    @patch.object(ProtT5Encoder, '_load_model')
    def test_encode_sequence_validation(self, mock_load_model, sequence_file, cache_dir):
        """Test that _encode_sequence validates inputs"""
        mock_load_model.return_value = None
        
        encoder = ProtT5Encoder(sequence_file, cache_dir)
        
        # Test with empty sequence
        with pytest.raises(ValueError, match="Sequence must be a non-empty string"):
            encoder._encode_sequence("")
        
        # Test with None
        with pytest.raises(ValueError, match="Sequence must be a non-empty string"):
            encoder._encode_sequence(None)
        
        # Test with whitespace-only sequence
        with pytest.raises(ValueError, match="Sequence is empty after cleaning"):
            encoder._encode_sequence("   ")
    
    @patch.object(ProtT5Encoder, '_load_model')
    def test_model_variant_options(self, mock_load_model, sequence_file, cache_dir):
        """Test that different ProtT5 model variants can be specified"""
        mock_load_model.return_value = None
        
        # Test standard model
        encoder1 = ProtT5Encoder(
            sequence_file,
            cache_dir,
            model_name="Rostlab/prot_t5_xl_uniref50"
        )
        assert encoder1.model_name == "Rostlab/prot_t5_xl_uniref50"
        
        # Test half-precision model
        encoder2 = ProtT5Encoder(
            sequence_file,
            cache_dir,
            model_name="Rostlab/prot_t5_xl_half_uniref50-enc"
        )
        assert encoder2.model_name == "Rostlab/prot_t5_xl_half_uniref50-enc"
        
        # Test XXL model
        encoder3 = ProtT5Encoder(
            sequence_file,
            cache_dir,
            model_name="Rostlab/prot_t5_xxl_uniref50"
        )
        assert encoder3.model_name == "Rostlab/prot_t5_xxl_uniref50"
    
    @patch.object(ProtT5Encoder, '_load_model')
    def test_device_auto_detection(self, mock_load_model, sequence_file, cache_dir):
        """Test automatic device detection"""
        mock_load_model.return_value = None
        
        # Test with None (should auto-detect)
        encoder1 = ProtT5Encoder(sequence_file, cache_dir, device=None)
        assert encoder1.device in ['cpu', 'cuda']
        
        # Test with 'auto' (should auto-detect)
        encoder2 = ProtT5Encoder(sequence_file, cache_dir, device='auto')
        assert encoder2.device in ['cpu', 'cuda']
        
        # Test with explicit 'cpu'
        encoder3 = ProtT5Encoder(sequence_file, cache_dir, device='cpu')
        assert encoder3.device == 'cpu'
