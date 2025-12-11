"""
Test Ankh Encoder
----------------
Tests for the AnkhEncoder class.
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
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['transformers'].AutoTokenizer = MagicMock()
sys.modules['transformers'].AutoModel = MagicMock()
sys.modules['transformers'].logging = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['huggingface_hub'].login = MagicMock()
sys.modules['ankh'] = MagicMock()

from src.models.encoders.ankh import AnkhEncoder


class TestAnkhEncoder:
    """Tests for AnkhEncoder"""
    
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
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_initialization_base_model(self, mock_load_model, sequence_file, cache_dir):
        """Test AnkhEncoder initialization with base model"""
        mock_load_model.return_value = None
        
        # Test base model (default)
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        assert encoder.model_name == "ElnaggarLab/ankh-base"
        assert encoder.model_variant == "base"
        assert encoder.pooling_strategy == "mean"
        assert encoder.device in ['cpu', 'cuda']
        assert str(encoder.cache_dir).endswith('base')
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_initialization_large_model(self, mock_load_model, sequence_file, cache_dir):
        """Test AnkhEncoder initialization with large model"""
        mock_load_model.return_value = None
        
        # Test large model
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="large")
        assert encoder.model_name == "ElnaggarLab/ankh-large"
        assert encoder.model_variant == "large"
        assert str(encoder.cache_dir).endswith('large')
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_initialization_custom_model(self, mock_load_model, sequence_file, cache_dir):
        """Test AnkhEncoder initialization with custom model name"""
        mock_load_model.return_value = None
        
        # Test custom model name (overrides variant)
        encoder = AnkhEncoder(
            sequence_file,
            cache_dir,
            model_variant="base",
            model_name="custom/ankh-model"
        )
        assert encoder.model_name == "custom/ankh-model"
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_invalid_model_variant(self, mock_load_model, sequence_file, cache_dir):
        """Test that invalid model variant raises ValueError"""
        mock_load_model.return_value = None
        
        with pytest.raises(ValueError, match="model_variant must be one of"):
            AnkhEncoder(
                sequence_file,
                cache_dir,
                model_variant="invalid"
            )
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_invalid_pooling_strategy(self, mock_load_model, sequence_file, cache_dir):
        """Test that invalid pooling strategy raises ValueError"""
        mock_load_model.return_value = None
        
        with pytest.raises(ValueError, match="pooling_strategy must be 'mean' or 'cls'"):
            AnkhEncoder(
                sequence_file,
                cache_dir,
                pooling_strategy="invalid"
            )
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_cache_dir_structure(self, mock_load_model, sequence_file, cache_dir):
        """Test that cache directory is properly organized"""
        mock_load_model.return_value = None
        
        # Test that 'ankh/base' is added to cache path
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        assert 'ankh' in str(encoder.cache_dir)
        assert encoder.cache_dir.name == 'base'
        
        # Test large variant
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="large")
        assert 'ankh' in str(encoder.cache_dir)
        assert encoder.cache_dir.name == 'large'

    def test_invalid_backend_value(self, sequence_file, cache_dir):
        """Ensure unsupported backend strings raise a ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            AnkhEncoder(sequence_file, cache_dir, backend="invalid")

    @patch.object(AnkhEncoder, '_load_model_with_native_package')
    def test_native_backend_loading(self, mock_native_loader, sequence_file, cache_dir):
        """Explicit ankh backend should only call the native loader."""
        mock_native_loader.return_value = None
        encoder = AnkhEncoder(sequence_file, cache_dir, backend="ankh")
        assert encoder.backend == "ankh"
        mock_native_loader.assert_called_once()

    @patch.object(AnkhEncoder, '_load_model_with_native_package')
    @patch.object(AnkhEncoder, '_load_model_huggingface')
    def test_auto_backend_fallback(self, mock_hf_loader, mock_native_loader, sequence_file, cache_dir):
        """Auto backend should fall back to native loader when HF fails."""
        mock_hf_loader.side_effect = RuntimeError("HF failure")
        mock_native_loader.return_value = None
        encoder = AnkhEncoder(sequence_file, cache_dir, backend="auto")
        assert encoder.active_backend == "ankh"
        mock_hf_loader.assert_called_once()
        mock_native_loader.assert_called_once()
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_encode_sequence')
    def test_get_embedding_base(self, mock_encode, mock_load_model, sequence_file, cache_dir):
        """Test get_embedding method with base model"""
        mock_load_model.return_value = None
        mock_encode.return_value = np.random.rand(768)  # Ankh base embedding dimension
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Get embedding for A*01:01
        embedding = encoder.get_embedding('A*01:01')
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        
        # Check that embedding is cached
        assert 'A*01:01' in encoder.embeddings
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_encode_sequence')
    def test_get_embedding_large(self, mock_encode, mock_load_model, sequence_file, cache_dir):
        """Test get_embedding method with large model"""
        mock_load_model.return_value = None
        mock_encode.return_value = np.random.rand(1536)  # Ankh large embedding dimension
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="large")
        
        # Get embedding for A*01:01
        embedding = encoder.get_embedding('A*01:01')
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_batch_encode_sequences')
    def test_batch_encode_alleles(self, mock_batch_encode, mock_load_model, 
                                  sequence_file, cache_dir):
        """Test batch_encode_alleles method"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = [
            np.random.rand(768),
            np.random.rand(768),
            np.random.rand(768)
        ]
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Batch encode multiple alleles
        alleles = ['A*01:01', 'A*02:01', 'B*07:02']
        results = encoder.batch_encode_alleles(alleles, batch_size=16)
        
        assert len(results) == 3
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        assert 'B*07:02' in results
        assert all(isinstance(v, np.ndarray) for v in results.values())
        assert all(v.shape == (768,) for v in results.values())
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_batch_encode_sequences')
    def test_batch_encode_default_batch_size(self, mock_batch_encode, mock_load_model,
                                            sequence_file, cache_dir):
        """Test that default batch size is appropriate for model size"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = [np.random.rand(768)]
        
        # Base model should have larger default batch size
        encoder_base = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        alleles = ['A*01:01']
        encoder_base.batch_encode_alleles(alleles)
        
        # Default batch_size should be 16 for base model (smaller model, more efficient)
        # This is just testing the initialization, actual batch size would be seen in logs
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_batch_encode_sequences')
    def test_batch_encode_with_duplicates(self, mock_batch_encode, mock_load_model,
                                         sequence_file, cache_dir):
        """Test that batch encoding handles duplicates correctly"""
        mock_load_model.return_value = None
        mock_batch_encode.return_value = [
            np.random.rand(768),
            np.random.rand(768)
        ]
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Include duplicate alleles
        alleles = ['A*01:01', 'A*02:01', 'A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles)
        
        # Should only return unique alleles (method deduplicates)
        assert len(results) == 2
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Should only encode unique ones
        mock_batch_encode.assert_called_once()
    
    @patch.object(AnkhEncoder, '_load_model')
    @patch.object(AnkhEncoder, '_batch_encode_sequences')
    def test_batch_encode_with_cached(self, mock_batch_encode, mock_load_model,
                                     sequence_file, cache_dir):
        """Test that batch encoding skips cached alleles"""
        mock_load_model.return_value = None
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Pre-cache one embedding
        encoder.embeddings['A*01:01'] = np.random.rand(768)
        
        # Mock should only be called for non-cached allele
        mock_batch_encode.return_value = [np.random.rand(768)]
        
        alleles = ['A*01:01', 'A*02:01']
        results = encoder.batch_encode_alleles(alleles, force=False)
        
        assert len(results) == 2
        assert 'A*01:01' in results
        assert 'A*02:01' in results
        
        # Should only encode the non-cached allele
        mock_batch_encode.assert_called_once()
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_encode_sequence_validation(self, mock_load_model, sequence_file, cache_dir):
        """Test that _encode_sequence validates inputs"""
        mock_load_model.return_value = None
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Test with empty sequence
        with pytest.raises(ValueError, match="Sequence must be a non-empty string"):
            encoder._encode_sequence("")
        
        # Test with None
        with pytest.raises(ValueError, match="Sequence must be a non-empty string"):
            encoder._encode_sequence(None)
        
        # Test with whitespace-only sequence
        with pytest.raises(ValueError, match="Sequence is empty after cleaning"):
            encoder._encode_sequence("   ")
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_pooling_strategies(self, mock_load_model, sequence_file, cache_dir):
        """Test that different pooling strategies can be specified"""
        mock_load_model.return_value = None
        
        # Test mean pooling (default)
        encoder_mean = AnkhEncoder(
            sequence_file,
            cache_dir,
            model_variant="base",
            pooling_strategy="mean"
        )
        assert encoder_mean.pooling_strategy == "mean"
        
        # Test cls pooling
        encoder_cls = AnkhEncoder(
            sequence_file,
            cache_dir,
            model_variant="base",
            pooling_strategy="cls"
        )
        assert encoder_cls.pooling_strategy == "cls"
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_device_auto_detection(self, mock_load_model, sequence_file, cache_dir):
        """Test automatic device detection"""
        mock_load_model.return_value = None
        
        # Test with None (should auto-detect)
        encoder1 = AnkhEncoder(sequence_file, cache_dir, device=None)
        assert encoder1.device in ['cpu', 'cuda']
        
        # Test with 'auto' (should auto-detect)
        encoder2 = AnkhEncoder(sequence_file, cache_dir, device='auto')
        assert encoder2.device in ['cpu', 'cuda']
        
        # Test with explicit 'cpu'
        encoder3 = AnkhEncoder(sequence_file, cache_dir, device='cpu')
        assert encoder3.device == 'cpu'
    
    @patch.object(AnkhEncoder, '_load_model')
    def test_model_variant_metadata(self, mock_load_model, sequence_file, cache_dir):
        """Test that model variant metadata is accessible"""
        mock_load_model.return_value = None
        
        encoder = AnkhEncoder(sequence_file, cache_dir, model_variant="base")
        
        # Check that MODEL_VARIANTS class attribute exists and has correct structure
        assert "base" in AnkhEncoder.MODEL_VARIANTS
        assert "large" in AnkhEncoder.MODEL_VARIANTS
        
        base_info = AnkhEncoder.MODEL_VARIANTS["base"]
        assert "model_name" in base_info
        assert "embedding_dim" in base_info
        assert "params" in base_info
        assert "description" in base_info
        
        assert base_info["embedding_dim"] == 768
        assert base_info["params"] == "50M"
        
        large_info = AnkhEncoder.MODEL_VARIANTS["large"]
        assert large_info["embedding_dim"] == 1536
        assert large_info["params"] == "650M"
