import os, random, numpy as np, torch, pandas as pd, json

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

def csv_merge(out_dir="./out_embs", output_csv="merged_embeddings.csv"):
    """
    Merges all embeddings (columns + cells) in an output directory into a single CSV file.

    Parameters
    ----------
    out_dir : str
        Path to the folder containing meta.json and .npy embeddings.
    output_csv : str
        Path to save the merged CSV file.
    """

    meta_path = os.path.join(out_dir, "meta.json")
    col_vecs_path = os.path.join(out_dir, "col_vecs.npy")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {out_dir}")
    if not os.path.exists(col_vecs_path):
        raise FileNotFoundError(f"col_vecs.npy not found in {out_dir}")

    # Load mapping and column embeddings
    with open(meta_path, "r") as f:
        meta = json.load(f)
    col_names = [c[0] for c in meta]

    col_vecs = np.load(col_vecs_path)
    d = col_vecs.shape[1]

    # Build DataFrame for column embeddings
    col_df = pd.DataFrame(col_vecs, columns=[f"dim_{i}" for i in range(d)])
    col_df.insert(0, "type", "column")
    col_df.insert(1, "name", col_names)
    col_df.insert(2, "row_index", np.arange(len(col_names)))

    # Append all cell-level embeddings (one .npy per column)
    all_rows = [col_df]

    for fname in os.listdir(out_dir):
        if fname.endswith("_embs.npy"):
            col_name = fname.replace("_embs.npy", "")
            emb_path = os.path.join(out_dir, fname)
            arr = np.load(emb_path)
            df = pd.DataFrame(arr, columns=[f"dim_{i}" for i in range(arr.shape[1])])
            df.insert(0, "type", "cell")
            df.insert(1, "name", col_name)
            df.insert(2, "row_index", np.arange(len(df)))
            all_rows.append(df)

    merged = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(out_dir, output_csv)
    merged.to_csv(csv_path, index=False)

    print(f"Saved merged embeddings to {csv_path} (shape {merged.shape})")
    return merged
