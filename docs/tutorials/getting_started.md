# Getting Started with HLA-ProtBERT

This tutorial will guide you through the basic usage of the HLA-ProtBERT library, demonstrating how to encode HLA alleles, analyze embeddings, and perform common tasks.

## Prerequisites

Before you begin, make sure you have:

1. Installed the HLA-ProtBERT package (see the [README.md](../../README.md) for installation instructions)
2. Downloaded and processed the IMGT/HLA database (using `scripts/update_imgt.py`)

## Setting Up

First, import the necessary modules and initialize the ProtBERT encoder:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import HLA-ProtBERT modules
from src.models.protbert import ProtBERTEncoder
from src.utils.logging import setup_logging

# Set up logging
logger = setup_logging(level="INFO")

# Set paths
data_dir = Path("./data")
sequence_file = data_dir / "processed" / "hla_sequences.pkl"
embeddings_dir = data_dir / "embeddings"

# Initialize encoder
encoder = ProtBERTEncoder(
    sequence_file=sequence_file,
    cache_dir=embeddings_dir,
    pooling_strategy="mean",
    use_peptide_binding_region=True
)

print(f"ProtBERT model: {encoder.model_name}")
print(f"Pooling strategy: {encoder.pooling_strategy}")
print(f"Using peptide binding region: {encoder.use_peptide_binding_region}")
print(f"Device: {encoder.device}")
```

## Exploring Available HLA Alleles

Let's examine what HLA alleles are available in our dataset:

```python
# Get allele counts by locus
alleles = list(encoder.sequences.keys())
print(f"Total number of alleles: {len(alleles)}")

# Count alleles per locus
locus_counts = {}
for allele in alleles:
    locus = allele.split('*')[0]
    locus_counts[locus] = locus_counts.get(locus, 0) + 1

# Display counts
for locus, count in sorted(locus_counts.items()):
    print(f"Locus {locus}: {count} alleles")

# Show a few example alleles for common loci
common_loci = ['A', 'B', 'C', 'DRB1']
for locus in common_loci:
    examples = [a for a in alleles if a.startswith(f"{locus}*")][:5]
    print(f"\nExample {locus} alleles: {', '.join(examples)}")
```

## Encoding Individual Alleles

Now let's encode some individual HLA alleles and examine their embeddings:

```python
# Define some common alleles to encode
test_alleles = ["A*01:01", "A*02:01", "B*07:02", "B*08:01", "C*07:01"]

# Encode each allele
embeddings = {}
for allele in test_alleles:
    try:
        # Get protein sequence
        sequence = encoder.get_sequence(allele)
        if sequence is None:
            print(f"No sequence found for {allele}")
            continue
        
        # Get embedding
        embedding = encoder.get_embedding(allele)
        embeddings[allele] = embedding
        
        # Print info
        print(f"\n{allele}:")
        print(f"  Sequence: {sequence[:50]}..." if len(sequence) > 50 else f"  Sequence: {sequence}")
        print(f"  Sequence length: {len(sequence)} amino acids")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    except Exception as e:
        print(f"Error encoding {allele}: {e}")
```

## Finding Similar Alleles

One of the key features of HLA-ProtBERT is the ability to find similar alleles based on embedding similarity:

```python
# Find similar alleles for each test allele
for allele in test_alleles:
    if allele not in embeddings:
        continue
        
    print(f"\nAlleles similar to {allele}:")
    similar = encoder.find_similar_alleles(allele, top_k=5)
    
    if not similar:
        print("  No similar alleles found")
        continue
        
    for similar_allele, similarity in similar:
        print(f"  {similar_allele}: similarity={similarity:.4f}")
```

## Comparing Multiple Alleles

Let's compare the similarity between different alleles:

```python
# Compare alleles
if len(embeddings) > 1:
    print("Pairwise similarities:")
    
    allele_list = list(embeddings.keys())
    for i, allele1 in enumerate(allele_list):
        for allele2 in allele_list[i+1:]:
            # Calculate cosine similarity
            similarity = encoder._cosine_similarity(embeddings[allele1], embeddings[allele2])
            print(f"  {allele1} vs {allele2}: {similarity:.4f}")
            
            # Interpret similarity
            if similarity > 0.99:
                print(f"    These alleles are identical or encode the same protein")
            elif similarity > 0.95:
                print(f"    These alleles are very similar (likely same protein group)")
            elif similarity > 0.90:
                print(f"    These alleles are functionally similar")
            elif similarity < 0.70:
                print(f"    These alleles are substantially different")
```

## Visualizing Embeddings with PCA

Let's visualize the embeddings to get a better understanding of the relationships between alleles:

```python
from sklearn.decomposition import PCA

# Get embeddings for specific loci
loci_to_visualize = ['A', 'B', 'C']
loci_alleles = {}
loci_embeddings = {}

for locus in loci_to_visualize:
    # Get alleles for this locus
    locus_alleles = [a for a in alleles if a.startswith(f"{locus}*")][:20]  # Limit to 20 alleles per locus
    loci_alleles[locus] = locus_alleles
    
    # Get embeddings
    locus_embeddings = [encoder.get_embedding(a) for a in locus_alleles]
    loci_embeddings[locus] = np.array(locus_embeddings)
    
    print(f"Got embeddings for {len(locus_alleles)} {locus} alleles")

# Combine all embeddings
all_embeddings = np.vstack([loci_embeddings[locus] for locus in loci_to_visualize])
all_labels = []
for locus in loci_to_visualize:
    all_labels.extend([locus] * len(loci_alleles[locus]))

# Apply PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 8))
colors = {'A': 'blue', 'B': 'red', 'C': 'green'}

for locus in loci_to_visualize:
    indices = [i for i, label in enumerate(all_labels) if label == locus]
    plt.scatter(
        reduced_embeddings[indices, 0], 
        reduced_embeddings[indices, 1],
        color=colors[locus],
        label=f"HLA-{locus}",
        alpha=0.7
    )

plt.title("PCA Visualization of HLA Allele Embeddings")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Creating a Similarity Heatmap

Let's create a heatmap to visualize similarity between common alleles:

```python
# Define common alleles for similarity matrix
common_alleles = [
    "A*01:01", "A*02:01", "A*03:01", "A*24:02",
    "B*07:02", "B*08:01", "B*15:01", "B*27:05",
    "C*01:02", "C*03:04", "C*07:01", "C*07:02"
]

# Calculate similarity matrix
similarity_matrix = np.zeros((len(common_alleles), len(common_alleles)))
valid_alleles = []

for i, allele1 in enumerate(common_alleles):
    try:
        embedding1 = encoder.get_embedding(allele1)
        valid_alleles.append(allele1)
        
        for j, allele2 in enumerate(common_alleles):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            elif j < i:
                similarity_matrix[i, j] = similarity_matrix[j, i]  # Symmetric
            else:
                try:
                    embedding2 = encoder.get_embedding(allele2)
                    similarity = encoder._cosine_similarity(embedding1, embedding2)
                    similarity_matrix[i, j] = similarity
                except Exception:
                    similarity_matrix[i, j] = np.nan
    except Exception:
        # Skip alleles that can't be encoded
        print(f"Skipping {allele1} - could not get embedding")
        continue

# Plot heatmap
if valid_alleles:  # Only plot if we have valid alleles
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix[:len(valid_alleles), :len(valid_alleles)],
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=valid_alleles,
        yticklabels=valid_alleles,
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title("Similarity Between Common HLA Alleles")
    plt.tight_layout()
    plt.show()
```

## Practical Use Case: Donor-Recipient Matching

Let's demonstrate a practical use case of HLA-ProtBERT for donor-recipient matching in transplantation:

```python
# Define donor and recipient HLA types
donor = {
    "A": ["A*01:01", "A*02:01"],
    "B": ["B*07:02", "B*08:01"],
    "C": ["C*07:01", "C*07:02"]
}

recipient = {
    "A": ["A*01:01", "A*24:02"],
    "B": ["B*07:02", "B*15:01"],
    "C": ["C*03:04", "C*07:01"]
}

# Function to calculate match score
def calculate_match_score(donor, recipient, encoder):
    match_scores = {}
    overall_match = 0.0
    count = 0
    
    for locus in donor.keys():
        locus_scores = []
        
        for d_allele in donor[locus]:
            for r_allele in recipient[locus]:
                try:
                    # Get embeddings
                    d_embedding = encoder.get_embedding(d_allele)
                    r_embedding = encoder.get_embedding(r_allele)
                    
                    # Calculate similarity
                    similarity = encoder._cosine_similarity(d_embedding, r_embedding)
                    locus_scores.append((d_allele, r_allele, similarity))
                    
                    overall_match += similarity
                    count += 1
                except Exception as e:
                    print(f"Error comparing {d_allele} and {r_allele}: {e}")
        
        match_scores[locus] = locus_scores
    
    # Calculate average match score
    avg_match = overall_match / count if count > 0 else 0.0
    
    return match_scores, avg_match

# Calculate match score
match_scores, avg_match = calculate_match_score(donor, recipient, encoder)

# Print results
print(f"Donor-Recipient HLA Match Analysis")
print(f"Overall Match Score: {avg_match:.4f}\n")

for locus, scores in match_scores.items():
    print(f"HLA-{locus} Matching:")
    for d_allele, r_allele, similarity in scores:
        match_quality = "Exact match" if similarity > 0.99 else \
                       "Very close match" if similarity > 0.95 else \
                       "Functional match" if similarity > 0.90 else \
                       "Partial match" if similarity > 0.80 else \
                       "Mismatch"
        print(f"  {d_allele} vs {r_allele}: {similarity:.4f} ({match_quality})")
    print()
```

## Batch Processing

For processing large numbers of alleles efficiently, you can use the batch encoding functionality:

```python
# Get all alleles for a specific locus
locus = "A"
locus_alleles = [a for a in alleles if a.startswith(f"{locus}*")][:50]  # Process first 50 alleles

# Get sequences
sequences = [encoder.get_sequence(a) for a in locus_alleles]
sequences = [s for s in sequences if s is not None]  # Filter out None values

# Batch encode
batch_embeddings = encoder.batch_encode(sequences, batch_size=8)
print(f"Batch encoded {len(sequences)} sequences")
print(f"Embeddings shape: {batch_embeddings.shape}")
```

## Next Steps

Now that you've learned the basics of using HLA-ProtBERT, you might want to explore:

1. **Fine-tuning**: Customize the model for specific tasks by fine-tuning on your own data
2. **Clinical prediction**: Use embeddings for predicting clinical outcomes
3. **Advanced analysis**: Combine with other tools for more sophisticated analysis
4. **Integration**: Integrate HLA-ProtBERT into your own pipelines

Check out the other tutorials and examples in the documentation for more advanced usage!
