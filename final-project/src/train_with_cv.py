import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from models.extreme_event_forecaster import ExtremeEventForecaster
from utils.ts_cv import rolling_time_series_folds
from utils.metrics import compute_binary_metrics
from baselines import baseline_persistence_logits, pooled_feature_matrix, train_logistic_regression, logistic_logits


class RadarSequenceDataset(Dataset):
    def __init__(self, npz_path, channel_idx=None):
        data = np.load(npz_path)
        X = data["X"]  # [N, T, C, H, W]
        if channel_idx is not None:
            X = X[:, :, channel_idx, :, :]
        self.X = X
        self.y_global = data["y_global"].astype(np.float32)
        self.y_region = data["y_region"].astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        yg = self.y_global[idx]
        yr = self.y_region[idx]
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
            nbr = []
            if i > 0: nbr.append((i - 1) * W + j)
            if i < H - 1: nbr.append((i + 1) * W + j)
            if j > 0: nbr.append(i * W + (j - 1))
            if j < W - 1: nbr.append(i * W + (j + 1))
            for n in nbr:
                A[idx, n] = 1.0
                A[n, idx] = 1.0
    I = torch.eye(N, device=device)
    A_hat = A + I
    deg = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


@torch.no_grad()
def eval_model(model, loader, A_hat, device):
    model.eval()
    y_true = []
    logits = []
    for x, yg, yr in loader:
        x = x.to(device)
        yg = yg.to(device)
        g, r = model(x, A_hat)
        y_true.append(yg.detach().cpu().numpy())
        logits.append(g.squeeze(-1).detach().cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    logits = np.concatenate(logits, axis=0)
    return compute_binary_metrics(y_true, logits)


def train_one_fold(dataset, train_idx, val_idx, test_idx, device, epochs=12, batch=16, lr=1e-3):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch, shuffle=False, num_workers=0)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, yg, yr in train_loader:
            x = x.to(device)
            yg = yg.to(device)
            yr = yr.to(device)

            optimizer.zero_grad()
            g, r = model(x, A_hat)
            loss_g = criterion(g.squeeze(-1), yg)
            loss_r = criterion(r, yr)
            loss = loss_g + 0.5 * loss_r
            loss.backward()
            optimizer.step()

        val_metrics = eval_model(model, val_loader, A_hat, device)
        if val_metrics["bce"] < best_val:
            best_val = val_metrics["bce"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics = eval_model(model, test_loader, A_hat, device)
    return model, test_metrics


def run_baselines(npz_path, fold):
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = data["y_global"].astype(np.float32)

    X_train = X[fold.train_idx]
    y_train = y[fold.train_idx]
    X_test = X[fold.test_idx]
    y_test = y[fold.test_idx]

    pers_logits = baseline_persistence_logits(X_test)
    pers_metrics = compute_binary_metrics(y_test, pers_logits)

    F_train = pooled_feature_matrix(X_train)
    F_test = pooled_feature_matrix(X_test)
    w, b = train_logistic_regression(F_train, y_train)
    lr_logits = logistic_logits(F_test, w, b)
    lr_metrics = compute_binary_metrics(y_test, lr_logits)

    return {"persistence": pers_metrics, "logistic": lr_metrics}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = os.path.join("data", "radar_sequences.npz")

    channel_idx = None
    dataset = RadarSequenceDataset(npz_path, channel_idx=channel_idx)
    n = len(dataset)

    folds = rolling_time_series_folds(
        n=n,
        train_size=max(120, int(0.6 * n)),
        val_size=max(30, int(0.15 * n)),
        test_size=max(30, int(0.15 * n)),
        step=max(30, int(0.1 * n)),
    )
    if len(folds) == 0:
        raise RuntimeError("Not enough data for the chosen fold sizes.")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    all_results = []
    for i, fold in enumerate(folds):
        model, metrics = train_one_fold(dataset, fold.train_idx, fold.val_idx, fold.test_idx, device)

        ckpt_path = os.path.join("checkpoints", f"forecaster_fold{i}.pt")
        torch.save({"model_state": model.state_dict()}, ckpt_path)

        base = run_baselines(npz_path, fold)

        out = {
            "fold": i,
            "n_train": len(fold.train_idx),
            "n_val": len(fold.val_idx),
            "n_test": len(fold.test_idx),
            "model": metrics,
            "baselines": base,
        }
        all_results.append(out)
        print(f"Fold {i}: model AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f} BCE={metrics['bce']:.4f}")

    out_path = os.path.join("results", "cv_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
