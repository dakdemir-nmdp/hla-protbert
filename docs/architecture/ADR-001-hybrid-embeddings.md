# ADR-001: Hybrid HLA Embeddings with Structure and MSA Information

**Status**: Proposed  
**Date**: 2025-01-19  
**Decision Makers**: HLA-ProtBERT Development Team  
**Supersedes**: N/A  

## Context and Problem Statement

Current HLA-ProtBERT encoders (ProtBERT, ESM-2, ProtT5, Ankh) provide state-of-the-art sequence-based embeddings. However, HLA alleles have rich additional information that could enhance embedding quality:

1. **3D Structural Information**: HLA alleles have known 3D structures from crystallography and AlphaFold predictions
2. **Multiple Sequence Alignments (MSA)**: IMGT/HLA database provides comprehensive MSAs showing evolutionary relationships
3. **Functional Annotations**: Peptide-binding region (PBR) annotations and serotype groupings

Recent research (MHCSeqNet2, ProtT5 studies) suggests that:
- Structure-aware embeddings improve binding prediction accuracy
- MSA information enhanced older models but shows mixed results with newer pLMs
- Hybrid approaches combining sequence + structure outperform sequence-only models for HLA-specific tasks

## Decision Drivers

- **Accuracy**: Improve downstream task performance (donor matching, peptide binding prediction)
- **Completeness**: Leverage all available HLA information, not just sequences
- **Modularity**: Maintain backward compatibility with existing sequence-only encoders
- **Research Alignment**: Follow best practices from recent protein language model research
- **Practical Utility**: Ensure new features provide measurable benefits for transplantation and immunology applications

## Considered Options

### Option 1: Structure-Aware Encoder (Priority 1)

Create a new `StructureAwareEncoder` that combines:
- Sequence embeddings (from existing encoders)
- Structural embeddings (C-alpha distance matrices, contact maps)
- Spatial attention mechanisms

**Implementation Approach**:
```python
class StructureAwareEncoder(HLAEncoder):
    """Hybrid encoder combining sequence and structure information.
    
    Architecture:
    1. Sequence embedding (ProtBERT/ESM/ProtT5/Ankh)
    2. Structure embedding (from PDB or AlphaFold)
    3. Attention-based fusion layer
    4. Final hybrid embedding
    """
    
    def __init__(
        self,
        sequence_encoder: str = "esm",  # Base sequence encoder
        structure_source: str = "alphafold",  # 'pdb', 'alphafold', or 'both'
        pbr_positions: tuple = (24, 301),  # Focus on peptide-binding region
        fusion_strategy: str = "attention",  # 'concat', 'attention', 'learned'
        **kwargs
    ):
        ...
    
    def _encode_sequence(self, sequence: str, structure_file: Optional[str] = None) -> np.ndarray:
        # 1. Get sequence embedding
        seq_embedding = self.sequence_encoder.get_embedding(sequence)
        
        # 2. Get structure embedding (if available)
        if structure_file:
            struct_embedding = self._encode_structure(structure_file)
        else:
            struct_embedding = self._predict_structure_embedding(sequence)
        
        # 3. Fuse embeddings
        hybrid_embedding = self._fuse_embeddings(seq_embedding, struct_embedding)
        
        return hybrid_embedding
```

**Data Sources**:
- **PDB Structures**: Download from RCSB PDB for known HLA structures
- **AlphaFold**: Use AlphaFold2/3 predictions for alleles without experimental structures
- **IMGT/HLA Alignments**: Map structural annotations to sequence positions

**Pros**:
- Leverages proven structural information
- Most HLA alleles have structural data or predictions
- Can improve donor matching by considering 3D compatibility
- Aligns with MHCSeqNet2 approach

**Cons**:
- Requires downloading/generating structure files
- Computational overhead for structure encoding
- Need to handle missing structures gracefully

### Option 2: MSA-Enhanced Encoder (Priority 2)

Enhance existing encoders with MSA information:

```python
class MSAEnhancedEncoder(HLAEncoder):
    """Encoder that incorporates multiple sequence alignment information.
    
    Uses MSA to:
    1. Provide evolutionary context
    2. Identify conserved vs variable regions
    3. Weight embedding by conservation scores
    """
    
    def __init__(
        self,
        base_encoder: str = "protbert",
        msa_source: str = "imgt",  # Use IMGT/HLA MSAs
        msa_weighting: str = "conservation",  # How to use MSA info
        **kwargs
    ):
        ...
    
    def _encode_sequence(self, sequence: str, allele: str) -> np.ndarray:
        # 1. Get base embedding
        base_embedding = self.base_encoder.get_embedding(sequence)
        
        # 2. Get MSA-derived features
        msa_features = self._extract_msa_features(allele)
        
        # 3. Weight embedding by conservation, variability, etc.
        enhanced_embedding = self._apply_msa_weighting(base_embedding, msa_features)
        
        return enhanced_embedding
```

**Data Sources**:
- **IMGT/HLA MSAs**: Available for all loci with comprehensive alignments
- **Conservation Scores**: Pre-compute using Shannon entropy or other metrics
- **Co-evolution Signals**: Identify residue pairs under selective pressure

**Pros**:
- Lightweight - no additional models needed
- MSAs readily available from IMGT/HLA
- Can highlight functionally important regions
- Proven benefit for older models (SeqVec, ProtBERT)

**Cons**:
- Recent research shows mixed results for newer pLMs (ProtT5, ESM-2)
- May not provide significant improvement over raw embeddings
- Need careful evaluation to confirm benefit

### Option 3: Full Hybrid Encoder (Priority 3)

Combine sequence, structure, and MSA into a unified model:

```python
class HybridHLAEncoder(HLAEncoder):
    """Complete hybrid encoder with sequence, structure, and MSA.
    
    Multi-modal architecture:
    - Sequence branch (ProtBERT/ESM/ProtT5/Ankh)
    - Structure branch (GNN or CNN on C-alpha graphs)
    - MSA branch (conservation profiles)
    - Cross-attention fusion layer
    """
    
    def __init__(
        self,
        sequence_encoder: str = "esm",
        use_structure: bool = True,
        use_msa: bool = True,
        fusion_architecture: str = "cross_attention",
        **kwargs
    ):
        ...
```

**Pros**:
- Maximum information utilization
- Modular - can disable components as needed
- Research-grade capability

**Cons**:
- Complex implementation
- High computational cost
- Risk of overfitting on HLA-specific tasks
- Difficult to maintain and extend

## Decision Outcome

**Chosen Option**: Phased implementation starting with **Option 1 (Structure-Aware Encoder)**

### Rationale

1. **Immediate Value**: Structure information has proven benefits for HLA tasks (MHCSeqNet2 results)
2. **Data Availability**: Most HLA alleles have structures (PDB) or high-quality predictions (AlphaFold)
3. **Modularity**: Can layer on top of existing encoders without breaking backward compatibility
4. **Research Trends**: Aligns with current protein language model + structure fusion approaches

### Implementation Roadmap

#### Phase 1: Structure-Aware Encoder (3-6 months)
- [ ] Download and organize HLA structure database (PDB + AlphaFold)
- [ ] Implement structure feature extraction (C-alpha distances, contact maps)
- [ ] Create `StructureAwareEncoder` class inheriting from `HLAEncoder`
- [ ] Implement fusion strategies (concatenation, attention-based)
- [ ] Benchmark on donor matching and peptide binding tasks
- [ ] Document usage and performance improvements

#### Phase 2: MSA Enhancement (optional, based on Phase 1 results)
- [ ] Pre-compute MSA-derived features for all IMGT/HLA alleles
- [ ] Implement `MSAEnhancedEncoder` or add MSA support to `StructureAwareEncoder`
- [ ] Evaluate benefit vs computational cost
- [ ] Document when MSA enhancement is recommended

#### Phase 3: Full Hybrid (research/optional)
- [ ] Design multi-modal fusion architecture
- [ ] Implement and train on HLA-specific tasks
- [ ] Compare to ensemble approaches
- [ ] Consider publication if significant improvement observed

### Backward Compatibility

All new encoders will:
- Inherit from `HLAEncoder` base class
- Support same API (`get_embedding`, `batch_encode_alleles`, etc.)
- Allow fallback to sequence-only mode if structure/MSA unavailable
- Cache embeddings using existing caching infrastructure

### Success Criteria

Structure-aware encoder is considered successful if:
1. **Accuracy Improvement**: ≥5% improvement on donor matching similarity metrics
2. **Coverage**: Works for ≥90% of IMGT/HLA alleles (via structures or predictions)
3. **Performance**: <2x slowdown vs base encoder
4. **Usability**: Can be used as drop-in replacement for existing encoders

## Consequences

### Positive

- **Enhanced accuracy** for transplantation and immunology applications
- **Research advantage**: State-of-the-art HLA embedding framework
- **Flexibility**: Users can choose sequence-only or hybrid encoders based on needs
- **Future-proof**: Architecture supports additional modalities (e.g., functional annotations)

### Negative

- **Complexity**: More moving parts to maintain
- **Data Management**: Need to manage structure databases (~10-50GB)
- **Compute Requirements**: Structure encoding requires more resources
- **Validation Burden**: Need comprehensive testing to ensure improvements are real

## Implementation Guidelines

### API Design

```python
# Sequence-only (existing)
from src.models.encoders import ESMEncoder
encoder = ESMEncoder("sequences.pkl")
embedding = encoder.get_embedding("A*01:01")  # 1280-dim

# Structure-aware (new)
from src.models.encoders import StructureAwareEncoder
encoder = StructureAwareEncoder(
    sequence_file="sequences.pkl",
    sequence_encoder="esm",  # Use ESM for sequence part
    structure_source="alphafold",  # Use AlphaFold structures
    fusion_strategy="attention"  # Attention-based fusion
)
embedding = encoder.get_embedding("A*01:01")  # 1280 + structure_dim

# Fallback when structure unavailable
embedding = encoder.get_embedding("A*99:99:99")  # Rare allele, no structure
# Returns sequence-only embedding with warning
```

### Data Organization

```
data/
├── structures/
│   ├── pdb/                    # Experimental structures
│   │   ├── A_01_01.pdb
│   │   └── ...
│   ├── alphafold/              # AlphaFold predictions
│   │   ├── A_01_01_predicted.pdb
│   │   └── ...
│   └── embeddings/             # Pre-computed structure embeddings
│       ├── A_01_01_struct.npy
│       └── ...
├── msa/                        # MSA data (if Phase 2)
│   ├── conservation/
│   └── alignments/
└── embeddings/
    ├── protbert/
    ├── esm/
    ├── prott5/
    ├── ankh/
    └── hybrid/                 # Hybrid embeddings
```

### Testing Requirements

1. **Unit Tests**: Each component tested in isolation
2. **Integration Tests**: Full pipeline from allele → hybrid embedding
3. **Performance Tests**: Measure speed, memory, accuracy
4. **Fallback Tests**: Ensure graceful degradation without structures
5. **Benchmark Suite**: Compare vs sequence-only on standard tasks

## References

- MHCSeqNet2 paper: Uses 3D structures with pairwise C-alpha distance matrices
- ProtT5 paper (Nature 2024): Shows MSA not always beneficial for newer pLMs
- AlphaFold HLA database: Comprehensive structure predictions for most alleles
- IMGT/HLA database: Gold-standard MSAs and sequence annotations

## Related Decisions

- ADR-002: Ensemble Embedding Strategies (to be written)
- ADR-003: Embedding Dimensionality Reduction for Production (to be written)

---

*This ADR follows the format from [Michael Nygard's ADR template](https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/templates/decision-record-template-by-michael-nygard/index.md)*
