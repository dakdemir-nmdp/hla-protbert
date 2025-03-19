# Contributing to HLA-ProtBERT

Thank you for your interest in contributing to HLA-ProtBERT! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Workflow](#workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a Code of Conduct that establishes how to engage with the community. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/YOUR-USERNAME/hla-protbert.git
   cd hla-protbert
   ```
3. **Set up a remote for the upstream repository**
   ```bash
   git remote add upstream https://github.com/dakdemir-nmdp/hla-protbert.git
   ```
4. **Create a virtual environment** and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

## Development Environment

We recommend the following development tools:

- **Code Formatter**: Black with line length of 88 characters
- **Linter**: Flake8
- **Import Sorter**: isort
- **Type Checker**: mypy
- **Editor**: VSCode with Python extension is recommended, but any editor works

## Workflow

1. **Create a branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits

3. **Write or update tests** for your changes

4. **Run the test suite** to ensure everything passes
   ```bash
   pytest
   ```

5. **Format and lint** your code
   ```bash
   black src tests examples
   flake8 src tests examples
   isort src tests examples
   ```

6. **Push your branch** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** from your fork to the main repository

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Include type hints for function parameters and return values
- Keep functions small and focused on a single task
- Write readable and maintainable code
- Add comments for complex logic
- Follow the DRY (Don't Repeat Yourself) principle

## Testing

- All new features should include tests
- All bug fixes should include tests that verify the fix
- Run the full test suite before submitting a PR
- Test coverage should not decrease with new code

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src
```

## Documentation

- Update documentation for any new features or changes to existing features
- Include docstrings for all modules, classes, and functions
- Update README.md if necessary
- Add examples for new features
- Follow a consistent style in your documentation

## Submitting Changes

1. **Ensure your branch is up to date** with the upstream repository
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a Pull Request (PR)** from your branch to the main repository
   - Provide a clear and descriptive title
   - Reference any related issues
   - Describe your changes in detail
   - Include any notes on dependencies or breaking changes
   - Make sure CI checks pass

3. **Address review comments** promptly
   - Make requested changes
   - Push updates to your branch
   - The PR will update automatically

4. **Once approved**, your PR will be merged by a maintainer

## Release Process

The release process is handled by the maintainers. We follow semantic versioning:

- **MAJOR.MINOR.PATCH**
  - **MAJOR**: Incompatible API changes
  - **MINOR**: New features in a backward-compatible manner
  - **PATCH**: Bug fixes in a backward-compatible manner

## Questions and Support

If you have questions or need help, please:
- Open an issue for bugs or feature requests
- Use GitHub discussions for general questions

Thank you for contributing to HLA-ProtBERT!
