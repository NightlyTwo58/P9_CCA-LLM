# main.py
import argparse, os, numpy as np, torch, json
from tqdm import tqdm

from utils import read_csv, sample_values_for_column, make_column_document, set_seed, csv_merge
from column_encoder import ColumnEncoder
from projection import Projection, project_numpy
from retrieval import FAISSIndex
from cell_encoder import CellEncoder
from num_normalize import normalize_number

# python -m src.main --csv data.csv --model_name facebook/opt-1.3b --cpu

def build_column_documents(df, sample_per_col=100, max_examples=50):
    docs, meta = [], []
    for col in df.columns:
        samples = sample_values_for_column(df, col, n=sample_per_col)
        doc = make_column_document(col, samples, max_examples=max_examples)
        docs.append(doc)
        meta.append((col, doc))
    return docs, meta

def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    csv_file_path = os.path.join(MAIN_DIR, "data", args.csv)
    if not os.path.exists(csv_file_path):
        print(f"Error: Specified CSV file not found at: {csv_file_path}")
        return

    print(f"Loading dataset: {csv_file_path}")
    df = read_csv(csv_file_path)

    print("Preparing column documents...")
    docs, meta = build_column_documents(df, args.sample_per_col, args.max_examples)

    # Load ColumnEncoder (LLM pooled embeddings)
    print(f"Loading LLM model: {args.model_name}")
    encoder = ColumnEncoder(
        model_name=args.model_name,
        device=("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"),
        use_peft=False
    )

    # Compute pooled column embeddings
    print("Embedding columns (this may take a while on CPU)...")
    col_vecs = np.array(
        encoder.batch_embed_texts(docs, pooling="mean", batch_size=args.embed_batch_size)
    ).astype(np.float32)
    print(f"Got column embeddings shape {col_vecs.shape}")

    # Optional projection
    proj_path = os.path.join(args.out_dir, "proj.pt")
    if args.project_dim > 0:
        proj = Projection(col_vecs.shape[1], args.project_dim)
        if os.path.exists(proj_path):
            proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
        col_vecs = project_numpy(col_vecs, proj, device="cpu", batch=64).astype(np.float32)
        print(f"Projected to {col_vecs.shape[1]} dims")

    # Build FAISS index
    retriever = FAISSIndex(dim=col_vecs.shape[1])
    retriever.add(col_vecs, meta)

    # Cell encoder with retrieval augmentation
    cell_encoder = CellEncoder(column_encoder=encoder, retriever=retriever, k=args.k)

    # Process columns
    for col, doc in tqdm(meta, desc="Encoding columns"):
        samples = sample_values_for_column(df, col, n=args.cells_per_col)
        # numeric normalization
        samples_norm = [str(normalize_number(s)) if normalize_number(s) is not None else s for s in samples]
        if not samples_norm:
            continue
        vecs = cell_encoder.batch_encode_cells(col, doc, samples_norm, topk=args.k)
        np.save(os.path.join(args.out_dir, f"{col}_embs.npy"), vecs.astype("float32"))

    # Save metadata
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("All embeddings saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_name", default="tiiuae/mistral-7b")
    parser.add_argument("--sample_per_col", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--embed_batch_size", type=int, default=2)
    parser.add_argument("--cells_per_col", type=int, default=20)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out_dir", default="./out_embs")
    parser.add_argument("--project_dim", type=int, default=256)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
    csv_merge("./out_embs", "merged_embeddings.csv")
