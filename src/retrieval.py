# retrieval.py
import faiss
import numpy as np
from typing import List, Tuple

class FAISSIndex:
    def __init__(self, dim: int, use_gpu=False):
        self.dim = dim
        self.use_gpu = use_gpu
        # simple L2 index
        self.index = faiss.IndexFlatIP(dim)  # use inner product with normalized vectors for cosine
        self.meta = []  # store (column_name, doc_text) for each vector

    def add(self, vectors: np.ndarray, metas: List[Tuple[str, str]]):
        # vectors must be float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.meta.extend(metas)

    def search(self, qvec: np.ndarray, topk=5):
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32)
        D, I = self.index.search(qvec.reshape(1, -1), topk)
        results = []
        for i in I[0]:
            if i == -1: continue
            results.append(self.meta[i])
        return results

    def batch_search(self, qvecs: np.ndarray, topk=5):
        if qvecs.dtype != np.float32:
            qvecs = qvecs.astype(np.float32)
        D, I = self.index.search(qvecs, topk)
        out = []
        for row in I:
            rec = []
            for i in row:
                if i == -1: continue
                rec.append(self.meta[i])
            out.append(rec)
        return out
