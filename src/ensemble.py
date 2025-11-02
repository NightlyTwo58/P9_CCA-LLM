import numpy as np

def ensemble_embeddings(list_of_emb_arrays):
    stacked = np.stack(list_of_emb_arrays, axis=0)
    avg = stacked.mean(axis=0)
    avg /= np.linalg.norm(avg, axis=1, keepdims=True).clip(min=1e-9)
    return avg
