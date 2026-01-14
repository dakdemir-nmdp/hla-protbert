#!/usr/bin/env python3
"""
Download all encoder models for HLA-ProtBERT.
This script downloads ProtBERT, ESM-2, ProtT5, Ankh Base, and Ankh Large models.
"""

from transformers import AutoModel, AutoTokenizer

print("\nStep 3: Downloading encoder models...")
print("=" * 60)

models = [
    ("ProtBERT", "Rostlab/prot_bert", "420M params"),
    ("ESM-2", "facebook/esm2_t33_650M_UR50D", "650M params"),
    ("ProtT5", "Rostlab/prot_t5_xl_uniref50", "1.3B params"),
    ("Ankh Base", "ElnaggarLab/ankh-base", "50M params"),
    ("Ankh Large", "ElnaggarLab/ankh-large", "650M params"),
]

for idx, (name, model_id, size) in enumerate(models, 1):
    print(f"\n{idx}. Downloading {name} ({size})...")
    try:
        if "prot_bert" in model_id:
            AutoTokenizer.from_pretrained(model_id, do_lower_case=False)
        else:
            AutoTokenizer.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
        print(f"✓ {name} downloaded")
    except Exception as e:
        print(f"✗ {name} failed: {e}")

print("\n✓ All models downloaded successfully!")
print("=" * 60)
