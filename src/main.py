# main.py
import argparse
import numpy as np
import os
from .utils import read_csv, sample_values_for_column, make_column_document
from .column_encoder import ColumnEncoder
from .retrieval import FAISSIndex
from .cell_encoder import CellEncoder
from .train_lora import train_lora

def build_column_documents(df, sample_per_col=100, max_examples=50):
    docs = []
    meta = []  # (col_name, doc_text)
    for col in df.columns:
        samples = sample_values_for_column(df, col, n=sample_per_col)
        doc = make_column_document(col, samples, max_examples=max_examples)
        docs.append(doc)
        meta.append((col, doc))
    return docs, meta

def main(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    csv_file_path = os.path.join(MAIN_DIR, "data", args.csv)

    if not os.path.exists(csv_file_path):
        print(f"Error: Specified CSV file not found at: {csv_file_path}")
        return

    df = read_csv(csv_file_path)

    docs, meta = build_column_documents(df, sample_per_col=args.sample_per_col, max_examples=args.max_examples)

    model_name = args.model_name
    peft_dir = args.peft_dir

    # If user chooses to fine-tune, run minimal LoRA training
    if args.fine_tune:
        print("Starting LoRA fine-tuning on column documents (this may take time)...")
        train_lora(model_name, docs, peft_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        model_to_load = peft_dir
        use_peft = True
    else:
        model_to_load = model_name
        use_peft = False

    # load ColumnEncoder
    from transformers import AutoTokenizer
    from peft import LoraConfig
    # If we used PEFT, construct a LoraConfig placeholder for ColumnEncoder (its __init__ will call get_peft_model)
    peft_config = None
    if use_peft:
        # When loading a peft-saved model we typically just load_from_pretrained; ColumnEncoder supports get_peft_model in __init__ with peft_config
        # Simpler: just instantiate ColumnEncoder with use_peft=False and then .model = AutoModelForCausalLM.from_pretrained(peft_dir, device_map="auto")
        pass

    encoder = ColumnEncoder(model_to_load, use_peft=False)  # if you saved PEFT, you can load from peft_dir manually
    # Create column embeddings
    texts = [t for t in docs]
    print("Embedding columns...")
    col_vecs = np.array(encoder.batch_embed_texts(texts, pooling="mean", batch_size=args.embed_batch_size)).astype(np.float32)
    # build FAISS
    dim = col_vecs.shape[1]
    retriever = FAISSIndex(dim=dim)
    retriever.add(col_vecs, meta)
    # cell encoder
    cell_encoder = CellEncoder(column_encoder=encoder, retriever=retriever, k=args.k)

    # Example: compute embeddings for the first N cells of each column
    print("Encoding cell samples with retrieval-augmented context...")
    output = {}
    for col, doc in meta:
        col_samples = sample_values_for_column(df, col, n=args.cells_per_col)
        if not col_samples:
            continue
        vecs = cell_encoder.batch_encode_cells(col, doc, col_samples, topk=args.k)
        output[col] = {"samples": col_samples, "embs": vecs}
    # Save embeddings (numpy)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    for col, d in output.items():
        np.save(os.path.join(out_dir, f"{col}_embs.npy"), d["embs"])
    print(f"Saved embeddings into {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_name", default="tiiuae/mistral-7b", help="HF model to use")
    parser.add_argument("--peft_dir", default="./peft_adapter")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sample_per_col", type=int, default=200)
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--embed_batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cells_per_col", type=int, default=20)
    parser.add_argument("--out_dir", default="./out_embs")
    args = parser.parse_args()
    main(args)
