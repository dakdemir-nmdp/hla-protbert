Below is an organized summary of the ESM repository README that you provided. This document outlines the purpose, available models, usage instructions, and additional resources for Evolutionary Scale Modeling:

---

### Overview

- **Purpose:**  
  The repository provides code and pre-trained weights for transformer-based protein language models developed by the Meta Fundamental AI Research Protein Team (FAIR). These models include:
  - **ESM-2** for structure prediction and property inference.
  - **ESMFold** for end-to-end protein structure prediction.
  - Other models like **ESM-1v**, **ESM-IF1**, and **ESM-MSA-1b** for tasks such as variant effect prediction and inverse folding.

- **Updates and Releases:**  
  The README indicates multiple updates (e.g., April 2023 protein design preprints) and provides links to code examples and preprints that detail recent advancements.

---

### Main Models and Their Usage

- **ESM-2:**  
  - Recommended for predicting structure, function, and other protein properties.
  - Several variants are available (ranging from 6M to 15B parameters).

- **ESMFold:**  
  - Provides an end-to-end approach for structure prediction.
  - Comes in two versions (`esmfold_v1` and `esmfold_v0`), with a command line interface for bulk predictions.

- **Other Models:**  
  - **ESM-IF1:** Inverse folding model for predicting protein sequences from structures.
  - **ESM-1v:** Designed for zero-shot prediction of variant effects.
  - **ESM-MSA-1b:** Processes multiple sequence alignments (MSA) to extract structural information.

---

### Installation and Quick Start

- **Installation:**  
  The repository can be installed via pip:
  ```bash
  pip install fair-esm
  ```
  For the latest (bleeding-edge) version:
  ```bash
  pip install git+https://github.com/facebookresearch/esm.git
  ```
  To work with ESMFold, ensure you use Python ≤ 3.9 and install with:
  ```bash
  pip install "fair-esm[esmfold]"
  ```

- **Quick Start Example:**  
  The README provides a sample Python script that demonstrates:
  - Loading an ESM-2 model.
  - Converting a batch of protein sequences into token representations.
  - Extracting per-residue and per-sequence embeddings.
  - Visualizing self-attention maps with Matplotlib.

  The code snippet highlights key steps such as:
  ```python
  import torch
  import esm
  
  # Load ESM-2 model and its alphabet for tokenization
  model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  model.eval()  # disable dropout
  
  # Prepare data for inference
  data = [
      ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
      # additional sequences...
  ]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  
  # Compute representations and contacts
  with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=True)
  ```

---

### Additional Functionalities

- **ESMFold Structure Prediction:**  
  Detailed instructions are provided for predicting protein structures and even generating PDB files from sequence data.

- **Bulk Embedding Extraction:**  
  Command-line interfaces (`esm-extract`) allow for efficient extraction of embeddings from FASTA files.

- **CPU Offloading and Large Models:**  
  Guidance is given for loading large models using Fully Sharded Data Parallel (FSDP) with CPU offloading to mitigate GPU memory constraints.

- **Zero-shot Variant Prediction and Inverse Folding:**  
  The README includes pointers to example notebooks and scripts for performing variant effect predictions and designing protein sequences based on structure.

- **Datasets and Resources:**  
  It provides information on pre-training dataset splits, the ESM Metagenomic Atlas, and links to additional datasets and tools for structural analysis.

---

### Citations and Licensing

- **Citations:**  
  The document contains BibTeX entries to properly credit the associated research papers (e.g., Rives et al. 2019, Lin et al. 2022).

- **License:**  
  The source code is licensed under the MIT license, with specific terms provided for using the ESM Metagenomic Atlas data.

---

### Conclusion

This README is an extensive resource for anyone interested in using state-of-the-art protein language models. It covers everything from installation and quick-start examples to advanced topics like bulk processing and model fine-tuning.

If you have any specific questions about any section or require further guidance on how to implement or adapt these models for your projects, please let me know.




```python
import random

import torch
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


@torch.no_grad()
def inverse_folding_example():
    tokenizer = EsmSequenceTokenizer()
    encoder = ESM3_structure_encoder_v0("cuda")
    model = ESM3_sm_open_v0("cuda")

    chain = ProteinChain.from_rcsb("1utn", "A")
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cuda()
    plddt = plddt.cuda()
    residue_index = residue_index.cuda()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )
    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
    sequence = tokenizer.decode(sequence_tokens[0])
    print(sequence)


@torch.no_grad()
def conditioned_prediction_example():
    tokenizers = get_model_tokenizers()

    model = ESM3_sm_open_v0("cuda")

    # PDB 1UTN
    sequence = "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    tokens = tokenizers.sequence.encode(sequence)

    # Calculate the number of tokens to replace, excluding the first and last token
    num_to_replace = int((len(tokens) - 2) * 0.75)

    # Randomly select indices to replace, excluding the first and last index
    indices_to_replace = random.sample(range(1, len(tokens) - 1), num_to_replace)

    # Replace selected indices with 32
    assert tokenizers.sequence.mask_token_id is not None
    for idx in indices_to_replace:
        tokens[idx] = tokenizers.sequence.mask_token_id
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)

    function_annotations = [
        # Peptidase S1A, chymotrypsin family
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    function_tokens = tokenizers.function.tokenize(function_annotations, len(sequence))
    function_tokens = tokenizers.function.encode(function_tokens)

    function_tokens = function_tokens.cuda().unsqueeze(0)
    sequence_tokens = sequence_tokens.cuda().unsqueeze(0)

    output = model.forward(
        sequence_tokens=sequence_tokens, function_tokens=function_tokens
    )
    return sequence, output, sequence_tokens


@torch.no_grad()
def decode(sequence, output, sequence_tokens):
    # To save on VRAM, we load these in separate functions
    decoder = ESM3_structure_decoder_v0("cuda")
    function_decoder = ESM3_function_decoder_v0("cuda")
    function_tokenizer = EsmFunctionTokenizer()

    # Generally not recommended to just argmax the logits, decode iteratively!
    # For quick demonstration only:
    structure_tokens = torch.argmax(output.structure_logits, dim=-1)
    structure_tokens = (
        structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
        .where(sequence_tokens != 2, 4097)  # EOS
        .where(sequence_tokens != 31, 4100)  # Chainbreak
    )

    bb_coords = (
        decoder.decode(
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )["bb_pred"]
        .detach()
        .cpu()
    )

    chain = ProteinChain.from_backbone_atom_coordinates(
        bb_coords, sequence="X" + sequence + "X"
    )
    chain.infer_oxygen().to_pdb("hello.pdb")

    # Function prediction
    p_none_threshold = 0.05
    log_p = F.log_softmax(output.function_logits[:, 1:-1, :], dim=3).squeeze(0)

    # Choose which positions have no predicted function.
    log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
    p_none = torch.exp(log_p_nones).mean(dim=1)  # "Ensemble of <none> predictions"
    where_none = p_none > p_none_threshold  # (length,)

    log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
    function_token_ids = torch.argmax(log_p, dim=2)
    function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]

    predicted_function = function_decoder.decode(
        function_token_ids,
        tokenizer=function_tokenizer,
        annotation_threshold=0.1,
        annotation_min_length=5,
        annotation_gap_merge_max=3,
    )

    print("function prediction:")
    print(predicted_function["interpro_preds"].nonzero())
    print(predicted_function["function_keywords"])


if __name__ == "__main__":
    inverse_folding_example()

    sequence, output, sequence_tokens = conditioned_prediction_example()
    torch.cuda.empty_cache()
    # And then decode from tokenized representation to outputs:
    decode(sequence, output, sequence_tokens)
    ```


    Understood. I will provide a precise, step-by-step guide for using the FAIR ESM library inside a `virtualenv` to compute protein sequence embeddings using a supported ESM model. This will include installation, model loading, and feature extraction.

# Setting Up and Using FAIR ESM for Protein Sequence Embeddings

FAIR **Evolutionary Scale Modeling (ESM)** provides pre-trained Transformer models for proteins, allowing you to extract high-dimensional embeddings for each amino acid (per-token) and for whole sequences (per-sequence). This guide will walk through creating a clean Python environment, installing ESM and PyTorch, and using an ESM model (e.g. **ESM2 650M**) to compute embeddings from a FASTA file. We will also cover how to handle the outputs and troubleshoot common issues.

## 1. Create a Python Virtual Environment

It’s best to start with a fresh virtual environment to avoid dependency conflicts. ESM is compatible with Python 3.7–3.10, but **Python 3.9** is recommended for full compatibility (especially if you plan to use the ESMFold model) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=To%20use%20the%20ESMFold%20model%2C,nvcc)). Here’s how to create and activate a virtual environment using Python 3.9 (ensure Python 3.9 is installed on your system):

```bash
# Create a virtual environment named "esm_env" with Python 3.9
python3.9 -m venv esm_env

# Activate the virtual environment (Linux/Mac)
source esm_env/bin/activate

# (On Windows PowerShell, use: .\esm_env\Scripts\Activate.ps1)
```

After activation, your shell prompt should indicate you’re in the `esm_env` environment. This isolated environment will contain the specific versions of PyTorch and ESM we need.

## 2. Install PyTorch and FAIR ESM

**PyTorch** is a prerequisite for ESM ([fair-esm · PyPI](https://pypi.org/project/fair-esm/0.1.0/#:~:text=As%20a%20prerequisite%2C%20you%20must,detected)), so install it first. Choose a PyTorch build that matches your system (CPU or CUDA GPU). For example, to install PyTorch with CUDA 11.8 support on Linux:

```bash
pip install --upgrade pip  # upgrade pip for latest package support

# Install PyTorch (GPU-enabled, CUDA 11.8) and torchvision. 
# (Check https://pytorch.org for the correct command for your CUDA/OS)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

*If you don’t have a GPU or prefer CPU-only, you can simply do:* `pip install torch`. (PyTorch will automatically use CPU if no CUDA is available.)

With PyTorch in place, install the **FAIR ESM** library. The most stable method is via PyPI:

```bash
pip install fair-esm
```

This will install the ESM package (and any essential dependencies) from PyPI. The ESM repo recommends a one-line pip install for the latest release ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=You%20can%20use%20this%20one,the%20latest%20release%20of%20esm)). Alternatively, you can get the cutting-edge version directly from GitHub:

```bash
# (Optional) Install latest ESM directly from GitHub main branch
pip install git+https://github.com/facebookresearch/esm.git
```

Using the PyPI release is generally sufficient for computing embeddings in production. Once installed, you should be able to import the `esm` module in Python.

**Verify the installation:** Run `python -c "import esm; print(esm.__version__)"`. This should show a version (e.g., `0.x.y`) and confirm that the library is available.

> **Note:** If you plan to use **ESMFold** (3D structure prediction), there are extra dependencies. You would install with `pip install "fair-esm[esmfold]"`, but this requires an environment with **Python ≤ 3.9** and a working CUDA toolkit (`nvcc`) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=To%20use%20the%20ESMFold%20model%2C,nvcc)). For simply extracting sequence embeddings, the base installation is enough.

## 3. Load a Pre-trained ESM Model in Python

With ESM installed, you can load a pre-trained model. The ESM repository provides many models (ESM-1b, ESM-1v, ESM-2 of various sizes, etc.). Here we’ll use **ESM2-T33 650M** (33-layer, 650 million parameters, trained on UR50/D dataset) as an example. This model is a good balance of power and size. Loading a model will automatically download the weights if not cached.

Open a Python interpreter or create a script and use the `esm.pretrained` API:

```python
import torch
import esm

# Load the ESM2-T33 650M model and its alphabet
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

model.eval()  # Disable dropout for deterministic results ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=model%2C%20alphabet%20%3D%20esm,disables%20dropout%20for%20deterministic%20results))

# If a GPU is available, move the model to GPU for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

When you call a loading function like `esm2_t33_650M_UR50D()`, ESM will download the model weights (stored in `~/.cache/torch/hub/` by default) if not already present. The `alphabet` object contains the mapping for tokens (amino acids) to indices that the model expects.

We set `model.eval()` to ensure the model is in inference mode (no dropout) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=model%2C%20alphabet%20%3D%20esm,disables%20dropout%20for%20deterministic%20results)). If you have a CUDA-compatible GPU, we transfer the model to GPU. (Make sure your PyTorch installation is GPU-enabled and the device is available.)

**Choosing a different model:** You can load other models by calling the appropriate function in `esm.pretrained`. For example:
- `esm1b_t33_650M_UR50S()` – Original ESM-1b model (33 layers, 650M params).
- `esm1v_t33_650M_UR90S_1()` – ESM-1v (variant scoring models, 5 versions available).
- `esm2_t6_8M_UR50D()` – A smaller ESM-2 model (6 layers, 8M params) for testing.
- `esm_msa1b_t12_100M_UR50S()` – ESM-MSA (multiple sequence alignment transformer).

Each loader returns a `(model, alphabet)` pair. The usage pattern is the same; only the model architecture and sizes differ. Make sure your hardware can handle the model size (ESM2 650M will use a few GB of GPU RAM; larger models like ESM-2 15B are extremely memory intensive and require special handling such as model sharding).

## 4. Prepare Input Sequences from a FASTA File

ESM models accept protein sequences as input. We will read sequences from a FASTA file and convert them to the format the model expects. **FASTA format** typically contains a header line starting with `>` followed by lines of sequence. For example:

```
>seq1
MVHFTAEEKAAVTSLWSKMNVEEAGGEALG...
>seq2
GLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLF...
```

First, load the sequences in Python. You can use libraries like Biopython’s `SeqIO`, but we'll show a simple approach to avoid extra dependencies:

```python
fasta_path = "input_sequences.fasta"  # path to your FASTA file

# Read FASTA into a list of (header, sequence) tuples
sequences = []
with open(fasta_path) as fasta:
    name = None
    seq_lines = []
    for line in fasta:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        if line.startswith(">"):
            # Header line
            if name and seq_lines:
                # save the previous sequence
                sequences.append((name, "".join(seq_lines)))
            name = line[1:]  # drop '>'
            seq_lines = []
        else:
            # Sequence line
            seq_lines.append(line)
    # Don't forget the last sequence in the file
    if name and seq_lines:
        sequences.append((name, "".join(seq_lines)))

print(f"Loaded {len(sequences)} sequences from FASTA.")
```

This will produce a list `sequences` where each element is a tuple `(header, seq)` suitable for ESM’s batch converter. For example, `sequences[0]` might look like `("seq1", "MVHFTAEEKAAVTSLWSKM...")`. 

**Important:** Ensure sequences contain only valid amino acid letters that the model recognizes. The standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) are supported, plus `X` for unknown. Non-standard letters or whitespace in sequences may cause errors.

Also, be mindful of sequence length. **ESM models have a maximum sequence length (typically 1022 amino acids for ESM-1b/ESM-2)** ([ESM-2nv 650M | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv650m#:~:text=Input%20Type,automatically%20truncated%20to%20this%20length)). Sequences longer than this may be truncated or cause an error. If you have very long sequences, consider splitting them or using the `--truncation_seq_length` option in the ESM CLI (described later).

## 5. Tokenize Sequences and Compute Embeddings

Now we’ll convert the amino acid sequences into model inputs (tokens) and run the model to get embeddings. ESM provides an **alphabet** object (we got it as `alphabet` when loading the model) which includes a `get_batch_converter()` method. This returns a **batch converter** function that can take a list of `(name, seq)` tuples and convert them into PyTorch tensors of token indices.

Let's convert our sequences to tokens:

```python
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
batch_tokens = batch_tokens.to(device)  # move tokens to GPU if model is on GPU

print(batch_tokens.shape)  
# For example, torch.Size([N, L_max]) where N = number of sequences, L_max = length of longest sequence (padded)
```

The `batch_converter` does several things:
- It adds special tokens required by the model (such as beginning-of-sequence `<bos>` and end-of-sequence `<eos>` tokens).
- It pads sequences in the batch to the same length (the length of the longest sequence) with a padding token.
- It returns `batch_tokens`, a tensor of shape `[batch_size, max_seq_length]` with integer token IDs for each sequence. It also returns `batch_labels` and `batch_strs` which we don't need further here.

Now, run the model to get embeddings. We will ask the model to return representations from the final layer. ESM’s `forward` can return a dictionary of representations for specified layers via the `repr_layers` argument. For example, `repr_layers=[33]` for ESM2 33-layer model will give us the final layer’s embeddings. We'll also disable gradient calculations since this is inference:

```python
# Ensure model is in eval mode (already done) and compute embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
# 'results' is a dict with keys 'logits' (if classifier head present) and 'representations'
representations = results["representations"][33]  # final layer representations

print(representations.shape)
# e.g., torch.Size([N, L_max, 1280]) for esm2_t33_650M (hidden dimension is 1280)
```

The tensor `representations` contains the embeddings for each sequence in the batch. Its shape is `[N, L_max, D]` where:
- `N` = number of sequences in the batch.
- `L_max` = the length (in tokens) of the longest sequence, which includes padding and special tokens.
- `D` = embedding dimension (e.g., 1280 for ESM2 650M).

For each sequence, `representations[i]` is a matrix of size `[L_i, D]` (where `L_i` is that sequence’s length with padding). It includes embeddings for special tokens and padded positions as well. Specifically:
- Index 0 of each sequence is the beginning-of-sequence token embedding (BOS).
- The last index for each sequence up to `L_max` may correspond to end-of-sequence (EOS) or padding. 

We need to extract the real per-residue embeddings for each sequence and optionally a pooled embedding per sequence:
- **Per-token embeddings:** the vector for each amino acid in the sequence.
- **Per-sequence embedding:** one vector representing the whole protein (we can use the average of token embeddings, or another pooling strategy).

First, let’s get actual sequence lengths (excluding padding and special tokens). We can compute the length of each sequence in tokens (including BOS/EOS but not padding) by finding how many tokens are not padding in each row:

```python
# Calculate true lengths (including BOS/EOS) for each sequence in the batch
padding_idx = alphabet.padding_idx
batch_lens = (batch_tokens != padding_idx).sum(dim=1).tolist()  # list of lengths per sequence
```

Now we iterate over each sequence to get embeddings:
```python
per_token_embeddings = []  # list to store per-residue embeddings for each sequence
per_sequence_embeddings = []  # list to store one embedding per sequence

for i, (name, seq) in enumerate(sequences):
    seq_length = batch_lens[i]
    # The representation tensor for sequence i
    # Slice from index 1 to seq_length-1 to skip BOS (0) and EOS (last) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Generate%20per,1%5D.mean%280))
    # This gives embeddings for actual amino acids in the sequence.
    token_emb = representations[i, 1:seq_length-1, :].clone().cpu()  
    per_token_embeddings.append(token_emb)
    
    # Compute the average embedding over all amino acids (per-sequence embedding)
    seq_emb = token_emb.mean(dim=0).clone()  # mean over the sequence length dimension
    per_sequence_embeddings.append(seq_emb)
    
    print(f"{name}: sequence length {len(seq)} -> token embeddings shape {token_emb.shape}, sequence embedding shape {seq_emb.shape}")
```

Here we take `representations[i, 1:seq_length-1, :]` as the per-residue embeddings for sequence `i`, **skipping index 0 (BOS) and the last index (`seq_length-1`, which is the EOS)** ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Generate%20per,1%5D.mean%280)). We then average those vectors to get a single 1×D sequence embedding. We store embeddings on CPU (using `.cpu()`) to free GPU memory if needed.

**Explanation:** The ESM authors note that “token 0 is always a beginning-of-sequence token, so the first residue is token 1” ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Generate%20per,1%5D.mean%280)). They recommend averaging from token 1 up to token N (and excluding the final token if it’s an end-of-sequence marker) to get a per-sequence representation. This is equivalent to what the official extractor does with `--include mean` ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%2A%20%60,token%20supervision)). (Using the BOS embedding directly is not advised for pre-trained models ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=,token%20supervision)).)

At this point:
- `per_token_embeddings[i]` is a tensor of shape `[L, D]` containing the embedding for each amino acid in the *i*-th sequence.
- `per_sequence_embeddings[i]` is a tensor of shape `[D]` containing the averaged embedding for the *i*-th sequence.

You can now work with these embeddings for downstream tasks (e.g., compare sequence similarities, input to another model, etc.).

## 6. Save or Visualize the Embeddings

**Storing embeddings:** It’s often useful to save the embeddings for later analysis instead of recomputing every time. There are a few ways to do this:

- **PyTorch `.pt` files:** Save the tensors using `torch.save`. For example:
  ```python
  # Save all embeddings in a dictionary
  result_dict = {name: {"per_token": per_token_embeddings[i], "per_sequence": per_sequence_embeddings[i]} 
                 for i, (name, seq) in enumerate(sequences)}
  torch.save(result_dict, "esm_embeddings.pt")
  ```
  This will serialize the embeddings dictionary to a file. Later, you can load it with `data = torch.load("esm_embeddings.pt")`.

- **NumPy or CSV:** Convert embeddings to NumPy arrays and save as `.npy` or text:
  ```python
  import numpy as np
  for name, seq_emb in zip([name for name, _ in sequences], per_sequence_embeddings):
      np.savetxt(f"{name}_mean_embedding.csv", seq_emb.numpy(), delimiter=",")
  ```
  This writes each sequence’s mean embedding vector to a CSV file (one row). Be cautious with this approach if your embedding dimension is large, as CSV will be bulky (for ESM2, D=1280).

- **HDF5 or other formats:** For many sequences, a binary format like HDF5 (using `h5py`) or NPZ might be more efficient to store a collection of embeddings.

**Visualizing embeddings:** Because the embeddings are high-dimensional (e.g., 1280-D), direct visualization is non-trivial. However, you can:
- Compute similarity/distance between sequence embeddings (e.g., cosine similarity) to cluster proteins.
- Use dimensionality reduction (PCA, t-SNE, UMAP) on the per-sequence embeddings to project them into 2D or 3D for visualization of relationships.
- For per-token embeddings, you might visualize certain properties (the ESM model also produces a contact map prediction as `results["contacts"]` if `return_contacts=True` ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Extract%20per,33)), which can be visualized as in the ESM README).

For example, to quickly visualize the unsupervised contact predictions from the model (which correlate with 3D structure), you could use the `results["contacts"]` matrix returned when `return_contacts=True` and plot it (as shown in the ESM README) – but this is an advanced use beyond just embeddings.

## 7. Troubleshooting and Tips

When setting up and running ESM, you might encounter some common issues. Here are some troubleshooting tips:

- **Python version** – If you get installation errors for the ESMFold extras or `openfold`, ensure you are using Python 3.9 or lower ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=To%20use%20the%20ESMFold%20model%2C,nvcc)). If you only use embeddings, ESM should work on newer Python versions, but the safest bet is 3.9 to match the official environment.

- **CUDA/GPU issues** – If PyTorch can’t find your GPU or you see `cuda is not available`, make sure you installed the correct CUDA-enabled PyTorch wheel for your system. Check `torch.cuda.is_available()`. If it's False on a machine with a GPU, you likely installed the CPU-only version of PyTorch – reinstall with the proper `--extra-index-url` for CUDA or use Conda to get the GPU build. Also ensure your NVIDIA drivers are compatible with the CUDA version. If you encounter a runtime error about mismatched CUDA versions, you may need to install a different PyTorch binary matching your driver’s CUDA (or upgrade drivers).

- **Out-of-memory (OOM) errors** – Large models and long sequences can consume a lot of memory. If you get OOM on GPU:
  - Try using a smaller model (e.g., ESM2 35M or 150M) or shorten your batch (fewer sequences at once).
  - Reduce sequence length (truncate sequences longer than 1022 amino acids, since the model won’t use beyond that ([ESM-2nv 650M | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv650m#:~:text=Input%20Type,automatically%20truncated%20to%20this%20length))).
  - Use mixed precision (half-precision) by calling `model.half()` and `batch_tokens = batch_tokens.half()` before inference, which can cut memory usage in half. Ensure your GPU supports float16.
  - For extremely large models (like ESM-2 15B), consider using inference with model sharding or CPU offloading. The ESM repo provides an example using FairScale’s FSDP to offload parts of the model to CPU memory ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=CPU%20offloading%20for%20inference%20with,large%20models)). This is advanced, but necessary if GPU memory is insufficient.

- **Performance** – If running on CPU, embedding extraction will be much slower than on GPU. You can still do it (and the code above will work on CPU), but expect it to take seconds per sequence for large models. For many sequences, try to use a machine with a GPU or use batch processing with `esm-extract` CLI which is optimized in C++/vectorized operations.

- **ESM CLI usage** – As an alternative to writing your own script, ESM provides a convenient command-line tool to extract embeddings in bulk ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=The%20following%20commands%20allow%20the,2%20model)). For example:
  ```bash
  esm-extract esm2_t33_650M_UR50D input_sequences.fasta embeddings_output/ \
      --repr_layers 33 --include per_tok mean
  ```
  This will create an `embeddings_output/` directory with one `.pt` file per sequence (containing the requested embeddings) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=Directory%20,file)). You can `torch.load()` these files in Python. The `--include` flag supports choices like `per_tok` (per-residue embeddings) and `mean` (average per-sequence) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%2A%20%60,token%20supervision)). Using the CLI is convenient for large-scale embedding extraction and uses the same underlying code we stepped through.

- **Non-standard amino acids** – If your FASTA contains characters outside the 20 standard amino acids (like `B, Z, U, O` or lowercase letters), the batch converter may throw an error. You should replace uncommon symbols with `X` (unknown) or remove them. The ESM alphabet by default includes `<mask>` for masked language modeling and might have an unknown token, but not explicitly others.

- **Installation issues** – If `pip install fair-esm` fails, make sure you upgraded pip and that your environment can build any requirements. The `fair-esm` package is pure Python (aside from PyTorch), so it should install quickly. If you see an error related to `dllogger` or `openfold` when using the `[esmfold]` option, it means those extra packages failed (likely due to missing `nvcc` or a compiler). In that case, stick to the base installation or ensure you have an NVIDIA CUDA toolkit installed for ESMFold.

By following this guide, you should be able to set up the FAIR ESM repository in a virtual environment and obtain protein sequence embeddings for any input FASTA. These embeddings (per-token and per-sequence) can be invaluable features for downstream analyses in protein science ([DeepRank-GNN-esm: a graph neural network for scoring protein ...](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbad191/7511844#:~:text=DeepRank,prediction%2C%20design%2C%20and%20functional)) ([Generating structural embeddings from FaceBook ESM model](https://www.biostars.org/p/9524101/#:~:text=Generating%20structural%20embeddings%20from%20FaceBook,accessibility%2C%20and%20many%20other)). With a stable environment and the tips above, you can integrate ESM embeddings into research pipelines or production systems for protein modeling.

**Sources:**

1. Meta AI (FAIR) ESM GitHub – *Installation and Usage* ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=You%20can%20use%20this%20one,the%20latest%20release%20of%20esm)) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Extract%20per,33))  
2. FAIR ESM README – *Computing embeddings and interpreting outputs* ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%23%20Generate%20per,1%5D.mean%280)) ([GitHub - facebookresearch/esm: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/facebookresearch/esm#:~:text=%2A%20%60,token%20supervision))  
3. NVIDIA NGC Model Card – *ESM2 650M model specifications (max sequence length, etc.)*