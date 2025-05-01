# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install dependencies: `pip install -r requirements.txt`
- Install development dependencies: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Run tests with coverage: `pytest --cov=src`
- Run lint: `black src tests examples && flake8 src tests examples && isort src tests examples`
- Generate embeddings: `python scripts/generate_embeddings.py --encoder-type protbert --locus A --all`

## Code Style
- Follow PEP 8 style guide
- Line length: 88 characters (Black default)
- Docstrings: Google style with type hints
- Imports: Use absolute imports; organize with isort
- Formatting: Use Black
- Type annotations: Required for function parameters and return values
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use try/except with specific exceptions and logging
- Logging: Use the logging module with appropriate levels

## Project Structure
- Package code under `src/`
- Scripts in `scripts/`
- Examples in `examples/`
- Tests mirror package structure in `tests/`