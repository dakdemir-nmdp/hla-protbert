# Changes Summary

This document summarizes the changes made to unify the ESM3 to ESM encoder naming and standardize interfaces across the codebase.

## Files Removed
- Removed `ESM3_Instructions.md` file (no longer needed)

## Encoder Naming Consistency

- Renamed all occurrences of `ESM3Encoder` to `ESMEncoder` for consistency
- Updated imports in all scripts and examples
- Updated encoder selection parameters in command-line arguments 
- Updated documentation strings to refer consistently to "ESM" instead of "ESM3"

## Parameter Standardization

- Standardized encoder initialization parameters across all scripts
- Added consistent handling of Hugging Face token through config
- Made cache directory structure consistent using encoder-specific subdirectories
- Updated default model references to use the same format: `facebook/esm2_t33_650M_UR50D`

## Example Updates

- Updated usage examples in README.md to reflect new naming
- Fixed parameter descriptions for consistency between examples
- Ensured basic_encoding.py and donor_matching.py use the same patterns
- Improved output formatting for encoder diagnostics

## Documentation

- Added comprehensive docstrings explaining the ESM and ProtBERT encoders
- Updated directory structure documentation in README.md
- Clarified model loading processes
- Improved command-line help text for all scripts

## Testing

- Verified that all scripts are functioning correctly after changes
- Fixed syntax error in analyze_locus_embeddings.py

## Additional Improvements

- Created CLAUDE.md with guidance for AI assistants working with the code
- Added build/test commands and code style guidelines
- Standardized output directory structure for all encoders