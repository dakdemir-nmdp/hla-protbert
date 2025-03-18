Based on your requirements, I'll outline a comprehensive GitHub repository structure for an HLA-ProtBERT encoding system that supports clinical applications. This plan integrates the IMGT/HLA data management, ProtBERT encoding, and extensibility for prediction tasks.

# HLA-ProtBERT: A Repository for HLA Allele Representation Learning

## Repository Structure

```
hla-protbert/
├── data/
│   ├── raw/                  # IMGT/HLA database files
│   ├── processed/            # Preprocessed sequence data
│   └── embeddings/           # Cached embeddings
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── imgt_downloader.py  # Tools to download and update IMGT data
│   │   ├── imgt_parser.py      # Parse IMGT/HLA files
│   │   └── sequence_utils.py   # Sequence preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py          # HLAEncoder base class
│   │   ├── protbert.py         # ProtBERT implementation
│   │   └── predictors.py       # Clinical prediction models
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── matching.py         # Donor-recipient matching analysis
│   │   ├── visualization.py    # Embedding visualization tools
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration handling
│       └── logging.py          # Logging utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_encoding_examples.ipynb
│   ├── 03_matching_analysis.ipynb
│   └── 04_clinical_prediction.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_encoder.py
│   └── test_predictors.py
├── scripts/
│   ├── update_imgt.py          # Script to update IMGT database
│   ├── generate_embeddings.py  # Generate and cache embeddings
│   └── train_predictor.py      # Train a clinical predictor
├── examples/
│   ├── basic_encoding.py
│   ├── donor_matching.py
│   └── clinical_prediction.py
├── docs/
│   ├── api/                    # API documentation
│   ├── tutorials/              # Step-by-step tutorials
│   └── index.md                # Main documentation page
├── .github/
│   └── workflows/
│       └── tests.yml           # GitHub Actions for CI/CD
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Implementation Plan

### 1. IMGT/HLA Data Management

#### a. Automated Download and Updates

```python
# src/data/imgt_downloader.py
class IMGTDownloader:
    def __init__(self, data_dir='./data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def download_latest(self):
        """Download the latest version of IMGT/HLA database"""
        # Check current version
        current_version = self._get_current_version()
        
        # Get latest version from IMGT/HLA website
        latest_version = self._get_latest_version()
        
        if current_version != latest_version:
            logger.info(f"Updating IMGT/HLA from {current_version} to {latest_version}")
            # Download FTP data
            self._download_ftp_data()
            # Extract files
            self._extract_data()
            # Update version info
            self._update_version_info(latest_version)
        else:
            logger.info(f"IMGT/HLA database already at latest version {current_version}")
            
    def _get_current_version(self):
        """Get currently installed version"""
        version_file = self.data_dir / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return None
```

#### b. Efficient Parsing of FASTA Files

```python
# src/data/imgt_parser.py
class IMGTParser:
    def __init__(self, imgt_dir='./data/raw', output_dir='./data/processed'):
        self.imgt_dir = Path(imgt_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def parse_protein_sequences(self):
        """Parse protein sequences from IMGT/HLA FASTA files"""
        sequences = {}
        
        # Parse consolidated protein file if available
        hla_prot_file = self.imgt_dir / "hla_prot.fasta"
        if hla_prot_file.exists():
            for record in SeqIO.parse(hla_prot_file, "fasta"):
                allele_name = self._extract_allele_name(record.description)
                sequences[allele_name] = str(record.seq)
        else:
            # Parse individual locus files
            for fasta_file in (self.imgt_dir / "fasta").glob("*_prot.fasta"):
                locus = fasta_file.stem.split('_')[0]
                for record in SeqIO.parse(fasta_file, "fasta"):
                    allele_name = self._extract_allele_name(record.description)
                    sequences[allele_name] = str(record.seq)
        
        # Save processed sequences
        with open(self.output_dir / "hla_sequences.pkl", 'wb') as f:
            pickle.dump(sequences, f)
            
        return sequences
```

### 2. ProtBERT Encoding

#### a. Locus-Specific Encoders

```python
# src/models/encoder.py
class HLAEncoder:
    """Base class for HLA encoders"""
    
    def __init__(self, sequence_file, cache_dir="./data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load sequences
        with open(sequence_file, 'rb') as f:
            self.sequences = pickle.load(f)
            
        # Initialize PYARD
        self.ard = pyard.ARD()
        
    def get_sequence(self, allele):
        """Get sequence for an allele with fallbacks"""
        # Implementation as in original code
        pass

# src/models/protbert.py
class ProtBERTEncoder(HLAEncoder):
    """ProtBERT-based encoder for HLA alleles"""
    
    def __init__(
        self, 
        sequence_file, 
        cache_dir="./data/embeddings",
        model_name="Rostlab/prot_bert", 
        locus=None,
        device=None
    ):
        super().__init__(sequence_file, cache_dir)
        
        self.locus = locus
        self.model_name = model_name
        
        # Set cache file based on locus
        if locus:
            self.embedding_cache_file = self.cache_dir / f"hla_{locus}_embeddings.pkl"
        else:
            self.embedding_cache_file = self.cache_dir / "hla_embeddings.pkl"
            
        # Load model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load cached embeddings
        self.embeddings = self._load_embedding_cache()
```

#### b. Fine-tuning Support

```python
# src/models/protbert.py (continued)
def fine_tune(
    self, 
    train_data, 
    labels, 
    output_dir="./models", 
    epochs=3,
    batch_size=8,
    learning_rate=5e-5
):
    """Fine-tune ProtBERT on HLA data"""
    # Set model to training mode
    self.model.train()
    
    # Create dataset
    dataset = HLADataset(train_data, labels, self.tokenizer)
    
    # Setup training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Define trainer
    trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train model
    trainer.train()
    
    # Save fine-tuned model
    self.model.save_pretrained(f"{output_dir}/fine_tuned")
    self.tokenizer.save_pretrained(f"{output_dir}/fine_tuned")
    
    # Update cache
    self.embeddings = {}
    self._save_embedding_cache()
```

### 3. Clinical Prediction Framework

#### a. Extensible Base Predictor

```python
# src/models/predictors.py
class HLAPredictor:
    """Base class for HLA-based clinical predictors"""
    
    def __init__(
        self, 
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True
    ):
        self.encoder = encoder
        self.clinical_variables = clinical_variables or []
        self.model_type = model_type
        self.freeze_encoder = freeze_encoder
        
        # Initialize model based on type
        if model_type == "mlp":
            self.model = self._create_mlp_model()
        elif model_type == "lstm":
            self.model = self._create_lstm_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_mlp_model(self):
        """Create a simple MLP model"""
        # Calculate input dimensions based on HLA and clinical variables
        hla_dim = 1024  # ProtBERT embedding size
        clinical_dim = len(self.clinical_variables)
        input_dim = hla_dim + clinical_dim
        
        # Define model architecture
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        return model
```

#### b. Specialized Clinical Applications

```python
# src/models/predictors.py (continued)
class TransplantOutcomePredictor(HLAPredictor):
    """Predicts transplant outcomes based on HLA matching and clinical variables"""
    
    def __init__(
        self,
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True,
        outcome_type="survival"  # survival, gvhd, rejection
    ):
        self.outcome_type = outcome_type
        super().__init__(encoder, clinical_variables, model_type, freeze_encoder)
    
    def prepare_input(self, donor_alleles, recipient_alleles, clinical_data):
        """Prepare model input from donor and recipient HLA typing plus clinical data"""
        # Get embeddings for donor and recipient
        donor_embeddings = {}
        recipient_embeddings = {}
        
        for allele in donor_alleles:
            locus = allele.split('*')[0]
            if locus not in donor_embeddings:
                donor_embeddings[locus] = []
            donor_embeddings[locus].append(self.encoder.get_embedding(allele))
            
        for allele in recipient_alleles:
            locus = allele.split('*')[0]
            if locus not in recipient_embeddings:
                recipient_embeddings[locus] = []
            recipient_embeddings[locus].append(self.encoder.get_embedding(allele))
        
        # Calculate differences between donor and recipient per locus
        diffs = []
        for locus in set(donor_embeddings.keys()) & set(recipient_embeddings.keys()):
            for d_emb in donor_embeddings[locus]:
                for r_emb in recipient_embeddings[locus]:
                    diff = np.abs(d_emb - r_emb)
                    diffs.append(diff)
        
        # Average the differences
        hla_features = np.mean(diffs, axis=0) if diffs else np.zeros(1024)
        
        # Add clinical variables
        clinical_features = np.array([clinical_data.get(var, 0) for var in self.clinical_variables])
        
        # Combine features
        features = np.concatenate([hla_features, clinical_features])
        
        return torch.tensor(features, dtype=torch.float32)
```

### 4. Command-Line Tools and Scripts

#### a. Update IMGT Database

```python
# scripts/update_imgt.py
import argparse
from src.data.imgt_downloader import IMGTDownloader
from src.data.imgt_parser import IMGTParser

def main():
    parser = argparse.ArgumentParser(description="Update IMGT/HLA database")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--force", action="store_true", help="Force update even if up to date")
    args = parser.parse_args()
    
    # Download latest IMGT/HLA database
    downloader = IMGTDownloader(data_dir=f"{args.data_dir}/raw")
    downloader.download_latest(force=args.force)
    
    # Parse downloaded data
    parser = IMGTParser(
        imgt_dir=f"{args.data_dir}/raw", 
        output_dir=f"{args.data_dir}/processed"
    )
    parser.parse_protein_sequences()
    
    print("IMGT/HLA database updated successfully!")

if __name__ == "__main__":
    main()
```

#### b. Generate Embeddings

```python
# scripts/generate_embeddings.py
import argparse
import pandas as pd
from pathlib import Path
from src.models.protbert import ProtBERTEncoder

def main():
    parser = argparse.ArgumentParser(description="Generate ProtBERT embeddings for HLA alleles")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--allele_file", help="File with alleles to encode (CSV or TXT)")
    parser.add_argument("--locus", help="Generate embeddings for specific locus only")
    parser.add_argument("--all", action="store_true", help="Generate embeddings for all known alleles")
    parser.add_argument("--model", default="Rostlab/prot_bert", help="ProtBERT model name")
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = ProtBERTEncoder(
        sequence_file=f"{args.data_dir}/processed/hla_sequences.pkl",
        cache_dir=f"{args.data_dir}/embeddings",
        model_name=args.model,
        locus=args.locus
    )
    
    if args.all:
        # Get all alleles from sequences
        alleles = list(encoder.sequences.keys())
        if args.locus:
            alleles = [a for a in alleles if a.startswith(f"{args.locus}*")]
    elif args.allele_file:
        # Load alleles from file
        file_path = Path(args.allele_file)
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            # Assume first column contains alleles
            alleles = df.iloc[:, 0].tolist()
        else:
            # Assume text file with one allele per line
            with open(file_path, 'r') as f:
                alleles = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Must specify either --all or --allele_file")
        return
    
    # Generate embeddings
    for i, allele in enumerate(alleles):
        try:
            embedding = encoder.get_embedding(allele)
            print(f"[{i+1}/{len(alleles)}] Encoded {allele}: shape={embedding.shape}")
        except Exception as e:
            print(f"[{i+1}/{len(alleles)}] Error encoding {allele}: {str(e)}")
    
    print(f"Successfully generated embeddings for {len(encoder.embeddings)} alleles")

if __name__ == "__main__":
    main()
```

### 5. Example Usage

#### Basic Usage Example

```python
# examples/basic_encoding.py
from src.models.protbert import ProtBERTEncoder

# Initialize encoder
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings"
)

# Get embedding for a specific allele
embedding = encoder.get_embedding("A*01:01")
print(f"Embedding shape: {embedding.shape}")

# Find similar alleles
similar = encoder.find_similar_alleles("A*02:01", top_k=5)
print("Similar alleles to A*02:01:")
for allele, score in similar:
    print(f"  {allele}: similarity={score:.4f}")
```

#### Transplant Matching Example

```python
# examples/donor_matching.py
from src.models.protbert import ProtBERTEncoder
from src.analysis.matching import MatchingAnalyzer

# Initialize encoder and analyzer
encoder = ProtBERTEncoder(
    sequence_file="./data/processed/hla_sequences.pkl",
    cache_dir="./data/embeddings"
)

analyzer = MatchingAnalyzer(encoder)

# Define donor and recipient HLA typing
donor = ["A*01:01", "A*02:01", "B*07:02", "B*08:01", "C*07:01", "C*07:02"]
recipient = ["A*01:01", "A*24:02", "B*07:02", "B*15:01", "C*03:04", "C*07:01"]

# Analyze matching
results = analyzer.analyze_matching(donor, recipient)

# Print results
print("HLA Matching Analysis")
print("-" * 50)
print(f"Exact matches: {results['exact_matches']}")
print(f"Functional matches: {results['functional_matches']}")
print(f"Average similarity: {results['average_similarity']:.4f}")

if results['mismatches']:
    print("\nMismatches (Recipient vs best Donor match):")
    for r_allele, d_allele, score in results['mismatches']:
        print(f"  {r_allele} vs {d_allele}: similarity={score:.4f}")

# Generate report
analyzer.generate_report(donor, recipient, output_file="matching_report.pdf")
```

## Documentation

### README.md

```markdown
# HLA-ProtBERT

A comprehensive framework for encoding HLA alleles using ProtBERT and applying these embeddings to clinical prediction tasks.

## Features

- **Automated IMGT/HLA Database Management**: Easy downloading and updating of the latest HLA sequence data
- **ProtBERT Encoding**: Convert HLA alleles to high-dimensional protein embeddings
- **Locus-Specific Models**: Separate encoders for different HLA loci (A, B, C, DRB1, etc.)
- **Transplant Matching**: Advanced donor-recipient compatibility analysis
- **Clinical Prediction**: Extensible framework for predicting clinical outcomes
- **Efficient Caching**: Save time by caching both sequences and embeddings

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/hla-protbert.git
cd hla-protbert
pip install -e .
```

### Download IMGT/HLA Database

```bash
python scripts/update_imgt.py
```

### Generate Embeddings

```bash
# Generate embeddings for all HLA-A alleles
python scripts/generate_embeddings.py --locus A --all
```

### Basic Usage

```python
from hlaprotbert.models import ProtBERTEncoder

# Initialize encoder
encoder = ProtBERTEncoder()

# Get embedding for an allele
embedding = encoder.get_embedding("A*01:01")

# Find similar alleles
similar = encoder.find_similar_alleles("A*02:01", top_k=5)
```

## Documentation

For detailed documentation, see the [docs](./docs) directory or visit our [GitHub Pages](https://yourusername.github.io/hla-protbert).

## Citation

If you use this framework in your research, please cite:

```
@software{hla_protbert,
  author = {Your Name},
  title = {HLA-ProtBERT: A framework for encoding HLA alleles using ProtBERT},
  year = {2025},
  url = {https://github.com/yourusername/hla-protbert}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## Continuous Integration and Testing

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov
        pip install -e .
    - name: Test with pytest
      run: |
        pytest --cov=src tests/
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

## Future Extensions

1. **Web Interface**:
   - Develop a Flask/FastAPI web application
   - Create interactive visualizations of HLA embeddings
   - Provide a user-friendly interface for clinical prediction

2. **Advanced Analysis**:
   - Integrate epitope prediction for T-cell alloreactivity
   - Add KIR-HLA interaction analysis
   - Develop structural analysis based on embeddings

3. **Deployment**:
   - Docker containers for easy deployment
   - Cloud-based compute for large-scale analysis
   - Integration with clinical systems via HL7/FHIR

This comprehensive repository structure provides a solid foundation for HLA research using ProtBERT embeddings, with extensibility for clinical applications and prediction tasks. The modular design allows for easy updates to the IMGT/HLA database, separate encoding per locus, and integration of clinical variables for advanced prediction models.