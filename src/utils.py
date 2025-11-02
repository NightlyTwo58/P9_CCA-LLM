import os, random, numpy as np, torch, pandas as pd

def read_csv(path):
    return pd.read_csv(path, dtype=str)

def sample_values_for_column(df, col, n=100):
    vals = df[col].dropna().astype(str).tolist()
    if not vals:
        return []
    random.shuffle(vals)
    return vals[:min(n, len(vals))]

def make_column_document(header, samples, max_examples=50):
    samples = samples[:max_examples]
    samples = [s.replace("\n", " ").strip() for s in samples if s.strip()]
    doc = f"Column: {header}\nExamples:\n"
    for s in samples:
        doc += f"- {s}\n"
    return doc

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False