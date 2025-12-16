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
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import HLAEncoder

# Create a concrete implementation of HLAEncoder for testing
class MockHLAEncoder(HLAEncoder):
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
        encoder = MockHLAEncoder(sequence_file, cache_dir)
        assert len(encoder.sequences) == 3
        assert 'A*01:01' in encoder.sequences
        assert 'A*02:01' in encoder.sequences
        assert 'B*07:02' in encoder.sequences
    
    def test_get_embedding(self, sequence_file, cache_dir):
        """Test get_embedding method"""
        encoder = MockHLAEncoder(sequence_file, cache_dir)
        
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
        """Test allele standardization
        
        Only standard IMGT/HLA formats should be parsed automatically.
        Ambiguous formats like 'A0101' are NOT supported to prevent errors.
        """
        encoder = MockHLAEncoder(sequence_file, cache_dir, locus='A')
        
        # Test with HLA- prefix removal
        assert encoder._standardize_allele('HLA-A*01:01') == 'A*01:01'
        
        # Test with A*0101 format (missing colon)
        assert encoder._standardize_allele('A*0101') == 'A*01:01'
        
        # Test with pure digits ONLY when locus is known (safe because context is explicit)
        assert encoder._standardize_allele('0101') == 'A*01:01'
        
        # Ambiguous format 'A0101' is NOT parsed (could be A*01:01 or A*10:10:1)
        assert encoder._standardize_allele('A0101') == 'A0101'  # Returns as-is
    
    def test_batch_encode_alleles(self, sequence_file, cache_dir):
        """Test batch_encode_alleles method"""
        encoder = MockHLAEncoder(sequence_file, cache_dir)
        
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

    def test_verify_ssl_toggle(self, sequence_file, cache_dir, monkeypatch):
        """Verify that SSL toggling reconfigures the HTTP backend once."""
        import src.models.encoder as encoder_module

        calls = []

        def fake_configure_http_backend(backend_factory=None):
            calls.append(backend_factory)

        monkeypatch.setattr(encoder_module, "configure_http_backend", fake_configure_http_backend, raising=False)
        monkeypatch.setattr(encoder_module, "HF_HUB_AVAILABLE", True, raising=False)

        # Reset class-wide state for the test
        encoder_module.HLAEncoder._current_ssl_verification = None

        encoder = MockHLAEncoder(sequence_file, cache_dir, verify_ssl=False)
        assert encoder.verify_ssl is False
        assert len(calls) == 1
        assert callable(calls[0])

        encoder2 = MockHLAEncoder(sequence_file, cache_dir, verify_ssl=True)
        assert encoder2.verify_ssl is True
        assert len(calls) == 2
        assert calls[1] is None  # Default backend when re-enabling

        # Cleanup for next tests
        encoder_module.HLAEncoder._current_ssl_verification = None

    def test_ard_instantiation(self, sequence_file, cache_dir):
        """Integration test: Verify ARD is instantiated correctly

        This test ensures that when py-ard is available, ARD() is called correctly
        (not pyard.ARD()), fixing a NameError that would occur when py-ard is installed.

        Fixes high-priority bug: src/models/encoder.py line 198
        """
        import src.models.encoder as encoder_module

        # Create a mock ARD class
        mock_ard_instance = MagicMock()
        mock_ard_class = MagicMock(return_value=mock_ard_instance)

        # Patch at module level before initialization
        with patch.object(encoder_module, 'PYARD_AVAILABLE', True):
            # Also need to inject ARD into the module
            encoder_module.ARD = mock_ard_class
            try:
                # Initialize encoder - this should call ARD() correctly
                encoder = MockHLAEncoder(sequence_file, cache_dir)

                # Verify ARD was instantiated (not pyard.ARD)
                mock_ard_class.assert_called_once_with()
                assert encoder.ard == mock_ard_instance
            finally:
                # Cleanup
                if hasattr(encoder_module, 'ARD'):
                    delattr(encoder_module, 'ARD')

    def test_ard_not_available(self, sequence_file, cache_dir, monkeypatch):
        """Test that encoder works when py-ard is not installed"""
        import src.models.encoder as encoder_module

        # Simulate py-ard not being available
        monkeypatch.setattr(encoder_module, 'PYARD_AVAILABLE', False)

        # Initialize encoder - this should not fail
        encoder = MockHLAEncoder(sequence_file, cache_dir)

        # Verify ARD is None
        assert encoder.ard is None

    def test_ard_fallback_resolution(self, sequence_file, cache_dir):
        """Integration test: Verify ARD allele resolution fallback works correctly

        This test ensures that when an allele is not found directly, the encoder
        attempts ARD mapping if py-ard is available.
        """
        import src.models.encoder as encoder_module

        # Mock ARD with redux_gl method
        mock_ard = MagicMock()
        mock_ard.redux_gl.return_value = 'A*01:01'  # Map to an existing allele

        # Mock ARD class
        mock_ard_class = MagicMock(return_value=mock_ard)

        with patch.object(encoder_module, 'PYARD_AVAILABLE', True):
            # Inject ARD into the module
            encoder_module.ARD = mock_ard_class
            try:
                encoder = MockHLAEncoder(sequence_file, cache_dir)

                # Try to get a sequence for an allele that doesn't match 2-field fallback
                # but would be mapped by ARD to an existing allele
                # Use a variant that won't match 2-field: A*99:99 (doesn't exist)
                sequence = encoder.get_sequence('A*99:99')

                # Verify ARD was called for mapping
                # ARD should be called because direct lookup and 2-field fallback fail
                assert sequence is not None
                mock_ard.redux_gl.assert_called()
                # Verify it returned the sequence for A*01:01 (the ARD mapping target)
                assert sequence == encoder.sequences['A*01:01']
            finally:
                # Cleanup
                if hasattr(encoder_module, 'ARD'):
                    delattr(encoder_module, 'ARD')
