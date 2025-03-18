#!/usr/bin/env python
"""
Setup script for hlaprotbert package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hlaprotbert",
    version="0.1.0",
    author="Deniz Akdemir",
    author_email="dakdemir@nmdp.org",
    description="HLA allele representation learning using ProtBERT embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dakdemir-nmdp/hla-protbert.git",
    project_urls={
        "Bug Tracker": "https://github.com/dakdemir-nmdp/hla-protbert/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "torch>=1.7.0",
        "transformers>=4.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
        "biopython",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "isort",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
        "analysis": [
            "umap-learn",
            "matplotlib",
            "seaborn",
            "reportlab",
        ],
        "nomenclature": [
            "pyard>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hla-update-imgt=scripts.update_imgt:main",
            "hla-generate-embeddings=scripts.generate_embeddings:main",
            "hla-train-predictor=scripts.train_predictor:main",
        ],
    },
)
