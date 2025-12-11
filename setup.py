#!/usr/bin/env python
"""
Setup script for hlaprotbert package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hlaprotbert",
    version="0.2.0",
    author="Deniz Akdemir",
    author_email="dakdemir@nmdp.org",
    description="HLA allele encoding using protein language models (ProtBERT, ESM) for immunogenomics research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dakdemir-nmdp/hla-protbert.git",
    project_urls={
        "Bug Tracker": "https://github.com/dakdemir-nmdp/hla-protbert/issues",
        "Documentation": "https://github.com/dakdemir-nmdp/hla-protbert/tree/main/docs",
        "Source Code": "https://github.com/dakdemir-nmdp/hla-protbert",
        "Changelog": "https://github.com/dakdemir-nmdp/hla-protbert/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.7.0",
        "transformers>=4.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "biopython>=1.78",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "isort>=5.10.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.15.0"
        ],
        "gpu": [
            "torch>=1.7.0"
        ],
        "analysis": [
            "umap-learn>=0.5.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "reportlab>=3.5.0"
        ],
        "nomenclature": [
            "pyard>=0.2.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "hla-download-imgt=scripts.download_imgt_data:main",
            "hla-update-imgt=scripts.update_imgt:main",
            "hla-generate-embeddings=scripts.generate_embeddings:main",
            "hla-analyze-locus=scripts.analyze_locus_embeddings:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "HLA", "immunogenetics", "protein language models", 
        "ProtBERT", "ESM", "embeddings", "bioinformatics",
        "transplantation", "histocompatibility"
    ],
)
