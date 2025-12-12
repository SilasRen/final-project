import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.extreme_event_forecaster import ExtremeEventForecaster


class RadarSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]          # [N, T, C, H, W]
        self.y_global = data["y_global"]  # [N]
        self.y_region = data["y_region"]  # [N, N_regions]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]          # [T, C, H, W] ndarray
        yg = self.y_global[idx]  # scalar (np.float32)
        yr = self.y_region[idx]  # [N_regions] ndarray
        return (
            torch.from_numpy(x).float(),
            torch.tensor(yg, dtype=torch.float32),
            torch.from_numpy(yr).float(),
        )


def build_grid_adjacency(H, W, device):
    N = H * W
    A = torch.zeros(N, N, device=device)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * W + j)
            if i < H - 1:
                neighbors.append((i + 1) * W + j)
            if j > 0:
                neighbors.append(i * W + (j - 1))
            if j < W - 1:
                neighbors.append(i * W + (j + 1))
            for n in neighbors:
                A[idx, n] = 1.0
                A[n, idx] = 1.0
    I = torch.eye(N, device=device)
    A_hat = A + I
    deg = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = os.path.join("data", "radar_sequences.npz")

    dataset = RadarSequenceDataset(npz_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    x0, yg0, yr0 = dataset[0]
    T, C, H, W = x0.shape
    num_regions = H * W

    model = ExtremeEventForecaster(
        in_channels=C,
        hidden_dim=32,
        num_gcn_layers=2,
        gcn_hidden_dim=32,
        num_regions=num_regions,
    ).to(device)

    A_hat = build_grid_adjacency(H, W, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_global = 0.0
        total_region = 0.0

        for x, yg, yr in loader:
            x = x.to(device)                  # [B, T, C, H, W]
            yg = yg.to(device)                # [B]
            yr = yr.to(device)                # [B, N_regions]

            optimizer.zero_grad()

            global_logits, regional_logits = model(x, A_hat)

            loss_global = criterion(global_logits.squeeze(-1), yg)
            loss_region = criterion(regional_logits, yr)

            loss = loss_global + 0.5 * loss_region
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_global += loss_global.item() * bsz
            total_region += loss_region.item() * bsz

        avg_loss = total_loss / len(dataset)
        avg_g = total_global / len(dataset)
        avg_r = total_region / len(dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"loss={avg_loss:.4f} "
            f"global={avg_g:.4f} "
            f"region={avg_r:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "H": H,
            "W": W,
        },
        os.path.join("checkpoints", "forecaster.pt"),
    )
    print("Model saved to checkpoints/forecaster.pt")


if __name__ == "__main__":
    train()
