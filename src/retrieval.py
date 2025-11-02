import faiss, numpy as np
from typing import List, Tuple

class FAISSIndex:
    def __init__(self, dim: int, use_gpu=False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine (after L2 norm)
        self.meta = []

    def add(self, vectors: np.ndarray, metas: List[Tuple[str, str]]):
        vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.meta.extend(metas)

    def search(self, qvec: np.ndarray, topk=5, score_threshold=0.5):
        qvec = qvec.astype("float32").reshape(1, -1)
        D, I = self.index.search(qvec, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or score < score_threshold:
                continue
            results.append((self.meta[idx], float(score)))
        return results
