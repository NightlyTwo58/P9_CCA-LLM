import os
import json
import random
from typing import List

def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str)

def sample_values_for_column(df, col, n=100):
    vals = df[col].dropna().astype(str).tolist()
    if not vals:
        return []
    random.shuffle(vals)
    return vals[:min(n, len(vals))]

def make_column_document(header: str, samples: List[str], max_examples=50):
    # create a compact textual "document" representing the column
    samples = samples[:max_examples]
    samples = [s.replace("\n", " ").strip() for s in samples if s.strip()]
    doc = f"Column: {header}\nExamples:\n"
    for s in samples:
        doc += f"- {s}\n"
    return doc
