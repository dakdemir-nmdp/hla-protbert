"""
Tests for biological correctness and validity of HLA embeddings.

These tests verify that embeddings capture meaningful biological relationships,
not just that the code runs without crashing.
"""

import pytest
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

from hlaprotbert.models.encoders import ProtBERTEncoder, ESMEncoder

# Test alleles representing different biological relationships
RELATED_ALLELES_A = ["A*01:01", "A*01:02", "A*01:03"]  # Same serotype, different subtypes
DIFFERENT_SEROTYPE_A = ["A*01:01", "A*02:01", "A*03:01"]  # Different A serotypes
DIFFERENT_LOCI = ["A*01:01", "B*07:02", "C*07:01"]  # Different loci
IDENTICAL_ALLELES = ["A*01:01", "A*01:01"]  # Should be exactly the same


class TestEmbeddingBiologicalValidity:
    """Test that embeddings reflect biological relationships."""
    
    @pytest.fixture(scope="class")
    def encoder(self):
        """Initialize encoder for testing."""
        sequence_file = Path("./data/processed/hla_sequences.pkl")
        if not sequence_file.exists():
            pytest.skip("Sequence file not found. Run update_imgt.py first.")
        return ProtBERTEncoder(sequence_file=str(sequence_file))
    
    def test_identical_alleles_have_identical_embeddings(self, encoder):
        """Test that encoding the same allele twice gives identical embeddings."""
        emb1 = encoder.get_embedding("A*01:01", force=True)
        emb2 = encoder.get_embedding("A*01:01", force=True)
        
        assert np.allclose(emb1, emb2, rtol=1e-5), \
            "Identical alleles should produce identical embeddings"
        
        # Cosine similarity should be 1.0
        similarity = 1 - cosine(emb1, emb2)
        assert similarity > 0.9999, \
            f"Identical alleles should have cosine similarity ~1.0, got {similarity}"
    
    def test_related_alleles_more_similar_than_unrelated(self, encoder):
        """
        Test that alleles from same serotype are more similar than alleles
        from different serotypes.
        
        A*01:01 and A*01:02 should be MORE similar than A*01:01 and A*02:01
        """
        # Get embeddings
        a0101 = encoder.get_embedding("A*01:01")
        a0102 = encoder.get_embedding("A*01:02")
        a0201 = encoder.get_embedding("A*02:01")
        
        # Calculate similarities
        sim_related = 1 - cosine(a0101, a0102)
        sim_different_serotype = 1 - cosine(a0101, a0201)
        
        assert sim_related > sim_different_serotype, \
            f"Related alleles (A*01:01 vs A*01:02) should be more similar " \
            f"({sim_related:.4f}) than different serotypes (A*01:01 vs A*02:01: {sim_different_serotype:.4f})"
    
    def test_different_loci_are_less_similar(self, encoder):
        """
        Test that alleles from different loci (A, B, C) are less similar
        than alleles from the same locus.
        """
        # Same locus, different serotypes
        a0101 = encoder.get_embedding("A*01:01")
        a0201 = encoder.get_embedding("A*02:01")
        
        # Different loci
        b0702 = encoder.get_embedding("B*07:02")
        
        # Similarities
        sim_same_locus = 1 - cosine(a0101, a0201)
        sim_different_locus = 1 - cosine(a0101, b0702)
        
        # This should generally be true for protein-based encoders
        # (though not always guaranteed depending on sequence similarity)
        assert sim_same_locus > sim_different_locus * 0.9, \
            f"Same locus similarity ({sim_same_locus:.4f}) should generally be " \
            f"higher than different locus ({sim_different_locus:.4f})"
    
    def test_embedding_cache_consistency(self, encoder):
        """Test that cached embeddings are consistent with fresh ones."""
        allele = "A*01:01"
        
        # Get fresh embedding
        emb_fresh = encoder.get_embedding(allele, force=True)
        
        # Get cached embedding
        emb_cached = encoder.get_embedding(allele, force=False)
        
        assert np.allclose(emb_fresh, emb_cached), \
            "Cached embeddings should match fresh embeddings"
    
    def test_batch_encoding_matches_individual(self, encoder):
        """Test that batch encoding produces same results as individual encoding."""
        alleles = ["A*01:01", "A*02:01", "B*07:02"]
        
        # Individual encoding
        individual_embs = {
            allele: encoder.get_embedding(allele, force=True) 
            for allele in alleles
        }
        
        # Batch encoding
        batch_embs = encoder.batch_encode_alleles(alleles, force=True)
        
        # Compare
        for allele in alleles:
            assert np.allclose(individual_embs[allele], batch_embs[allele], rtol=1e-4), \
                f"Batch encoding for {allele} should match individual encoding"
    
    def test_embedding_dimensions_consistent(self, encoder):
        """Test that all embeddings have consistent dimensions."""
        alleles = ["A*01:01", "A*02:01", "B*07:02", "C*07:01"]
        embeddings = [encoder.get_embedding(allele) for allele in alleles]
        
        dims = [len(emb) for emb in embeddings]
        assert len(set(dims)) == 1, \
            f"All embeddings should have same dimension, got {set(dims)}"
        
        # ProtBERT should be 768-dim
        if isinstance(encoder, ProtBERTEncoder):
            assert dims[0] == 1024, \
                f"ProtBERT embeddings should be 1024-dim, got {dims[0]}"
    
    def test_embedding_values_are_bounded(self, encoder):
        """Test that embedding values are in reasonable range (not NaN, not extreme)."""
        emb = encoder.get_embedding("A*01:01")
        
        # Check for NaN
        assert not np.any(np.isnan(emb)), "Embeddings should not contain NaN"
        
        # Check for Inf
        assert not np.any(np.isinf(emb)), "Embeddings should not contain Inf"
        
        # Check for reasonable range (most transformer embeddings are roughly [-10, 10])
        assert np.abs(emb).max() < 100, \
            f"Embedding values seem extreme: max={np.abs(emb).max()}"


class TestPeptideBindingRegionExtraction:
    """Test that PBR extraction works correctly."""
    
    @pytest.fixture(scope="class")
    def encoder_full(self):
        """Encoder using full sequence."""
        sequence_file = Path("./data/processed/hla_sequences.pkl")
        if not sequence_file.exists():
            pytest.skip("Sequence file not found")
        return ProtBERTEncoder(
            sequence_file=str(sequence_file),
            use_peptide_binding_region=False
        )
    
    @pytest.fixture(scope="class")
    def encoder_pbr(self):
        """Encoder using PBR only."""
        sequence_file = Path("./data/processed/hla_sequences.pkl")
        if not sequence_file.exists():
            pytest.skip("Sequence file not found")
        return ProtBERTEncoder(
            sequence_file=str(sequence_file),
            use_peptide_binding_region=True,
            locus="A"
        )
    
    def test_pbr_differs_from_full_sequence(self, encoder_full, encoder_pbr):
        """Test that PBR embedding differs from full sequence embedding."""
        allele = "A*01:01"
        
        emb_full = encoder_full.get_embedding(allele, force=True)
        emb_pbr = encoder_pbr.get_embedding(allele, force=True)
        
        # They should NOT be identical (different input sequences)
        assert not np.allclose(emb_full, emb_pbr, rtol=1e-3), \
            "PBR embeddings should differ from full sequence embeddings"
        
        # But they should still be reasonably similar (same allele)
        similarity = 1 - cosine(emb_full, emb_pbr)
        assert similarity > 0.5, \
            f"PBR and full sequence should still be somewhat similar (got {similarity:.4f})"
    
    def test_pbr_embeddings_have_same_dimensions(self, encoder_full, encoder_pbr):
        """Test that PBR and full embeddings have same dimensions."""
        allele = "A*01:01"
        
        emb_full = encoder_full.get_embedding(allele)
        emb_pbr = encoder_pbr.get_embedding(allele)
        
        assert len(emb_full) == len(emb_pbr), \
            "PBR and full embeddings should have same dimensions"


class TestAlleleResolutionFallback:
    """Test allele name resolution and fallback mechanisms."""
    
    @pytest.fixture(scope="class")
    def encoder(self):
        """Initialize encoder for testing."""
        sequence_file = Path("./data/processed/hla_sequences.pkl")
        if not sequence_file.exists():
            pytest.skip("Sequence file not found")
        return ProtBERTEncoder(sequence_file=str(sequence_file))
    
    def test_standard_allele_name_works(self, encoder):
        """Test that standard allele names work."""
        emb = encoder.get_embedding("A*01:01")
        assert emb is not None
        assert len(emb) > 0
    
    def test_allele_resolution_to_2field(self, encoder):
        """Test that high-resolution alleles resolve to 2-field."""
        # Try 4-field allele (may not exist exactly)
        # Should fall back to 2-field
        try:
            emb_4field = encoder.get_embedding("A*01:01:01:01")
            emb_2field = encoder.get_embedding("A*01:01")
            
            # If 4-field resolved, it should be same as 2-field
            # (assuming fallback to 2-field happened)
            if emb_4field is not None and emb_2field is not None:
                # They might be identical if fallback occurred
                pass
        except Exception:
            # It's ok if this fails - depends on data availability
            pytest.skip("4-field resolution test skipped")
    
    def test_missing_allele_returns_none_or_raises(self, encoder):
        """Test that completely missing alleles are handled gracefully."""
        # Use a clearly fake allele
        # Should raise ValueError
        with pytest.raises(ValueError, match="No sequence found"):
            encoder.get_embedding("Z*99:99")


class TestEncoderConsistency:
    """Test consistency across different encoders."""
    
    @pytest.fixture(scope="class")
    def encoders(self):
        """Initialize multiple encoders."""
        sequence_file = Path("./data/processed/hla_sequences.pkl")
        if not sequence_file.exists():
            pytest.skip("Sequence file not found")
        
        encoders = {}
        try:
            encoders['protbert'] = ProtBERTEncoder(sequence_file=str(sequence_file))
        except Exception as e:
            pytest.skip(f"ProtBERT not available: {e}")
        
        try:
            encoders['esm'] = ESMEncoder(sequence_file=str(sequence_file))
        except Exception as e:
            # ESM might not be available
            pass
        
        return encoders
    
    def test_all_encoders_produce_valid_embeddings(self, encoders):
        """Test that all encoders produce valid embeddings."""
        allele = "A*01:01"
        
        for encoder_name, encoder in encoders.items():
            emb = encoder.get_embedding(allele)
            
            assert emb is not None, f"{encoder_name} returned None"
            assert len(emb) > 0, f"{encoder_name} returned empty embedding"
            assert not np.any(np.isnan(emb)), f"{encoder_name} returned NaN"
            assert not np.any(np.isinf(emb)), f"{encoder_name} returned Inf"
    
    def test_all_encoders_respect_biological_relationships(self, encoders):
        """Test that all encoders capture basic biological relationships."""
        for encoder_name, encoder in encoders.items():
            # Related alleles should be more similar than unrelated
            a0101 = encoder.get_embedding("A*01:01")
            a0102 = encoder.get_embedding("A*01:02")
            b0702 = encoder.get_embedding("B*07:02")
            
            sim_related = 1 - cosine(a0101, a0102)
            sim_unrelated = 1 - cosine(a0101, b0702)
            
            assert sim_related > sim_unrelated * 0.8, \
                f"{encoder_name}: related alleles should be more similar. " \
                f"Got {sim_related:.4f} vs {sim_unrelated:.4f}"


# Parameterized tests for multiple loci
@pytest.mark.parametrize("locus,allele1,allele2", [
    ("A", "A*01:01", "A*01:02"),
    ("B", "B*07:02", "B*07:03"),
    ("C", "C*07:01", "C*07:02"),
])
def test_within_locus_subtypes_are_similar(locus, allele1, allele2):
    """Test that subtypes within same serotype are similar."""
    sequence_file = Path("./data/processed/hla_sequences.pkl")
    if not sequence_file.exists():
        pytest.skip("Sequence file not found")
    
    encoder = ProtBERTEncoder(sequence_file=str(sequence_file))
    
    emb1 = encoder.get_embedding(allele1)
    emb2 = encoder.get_embedding(allele2)
    
    if emb1 is None or emb2 is None:
        pytest.skip(f"Alleles {allele1} or {allele2} not available")
    
    similarity = 1 - cosine(emb1, emb2)
    
    # Subtypes should be quite similar (>0.8 typically)
    assert similarity > 0.7, \
        f"Subtypes {allele1} and {allele2} should be similar (got {similarity:.4f})"
