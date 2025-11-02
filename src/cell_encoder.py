# cell_encoder.py
from typing import List
import numpy as np
import textwrap

class CellEncoder:
    """
    Given:
      - a ColumnEncoder that can embed text
      - a FAISSIndex of column embeddings with metadata (column_name, column_doc_text)
    This builds a retrieval-augmented prompt for each cell:
      Prompt := [target_column_doc] + [top-k retrieved column docs] + [cell_text]
    Then it gets an embedding from the LLM (same ColumnEncoder) by pooling.
    """
    def __init__(self, column_encoder, retriever, k=3, max_prompt_examples=50):
        self.column_encoder = column_encoder
        self.retriever = retriever
        self.k = k
        self.max_prompt_examples = max_prompt_examples

    def build_prompt_for_cell(self, column_header: str, column_doc: str, cell_text: str, retrieved_cols: List[tuple]):
        # retrieved_cols: list of (col_name, col_doc_text)
        prompt = f"Target column: {column_header}\n{column_doc}\n\n"
        if retrieved_cols:
            prompt += "Related columns and examples:\n"
            for name, doc in retrieved_cols[:self.k]:
                # shorten docs if very long
                doc_short = textwrap.shorten(doc, width=800, placeholder=" [...]")
                prompt += f"Column: {name}\n{doc_short}\n\n"
        prompt += f"Cell to encode: {cell_text}\n\nEncode the cell relative to the target column and the related columns."
        return prompt

    def encode_cell(self, column_header: str, column_doc: str, cell_text: str, topk=3):
        # embed the cell with retrieval-augmented prompt
        # get column doc vector (we could also use column embedding directly)
        cell = str(cell_text)
        # compute base embedding for cell-only for retrieval query: we can embed the cell alone
        base_q = self.column_encoder.embed_text(cell)
        retrieved = self.retriever.search(base_q.astype(np.float32), topk)
        prompt = self.build_prompt_for_cell(column_header, column_doc, cell, retrieved)
        vec = self.column_encoder.embed_text(prompt)
        return vec

    def batch_encode_cells(self, column_header: str, column_doc: str, cells: List[str], topk=3):
        out = []
        for cell in cells:
            v = self.encode_cell(column_header, column_doc, cell, topk=topk)
            out.append(v)
        return np.stack(out)
