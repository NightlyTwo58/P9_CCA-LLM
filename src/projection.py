import torch, torch.nn as nn, numpy as np

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def project_numpy(vectors: np.ndarray, proj: "Projection", device="cpu", batch=128):
    proj.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(vectors), batch):
            b = torch.tensor(vectors[i:i+batch], dtype=torch.float32).to(device)
            v = proj(b).cpu().numpy()
            norms = np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)
            outs.append(v / norms)
    return np.vstack(outs)
