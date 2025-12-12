import os
import numpy as np
import torch
import torch.nn.functional as F

from models.extreme_event_forecaster import ExtremeEventForecaster
from train_forecaster import RadarSequenceDataset, build_grid_adjacency


def apply_warming_scenario(x, temp_channel_idx, delta_T):
    x_s = x.clone()
    x_s[:, :, temp_channel_idx, :, :] += delta_T
    return x_s


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    values = np.asarray(values)
    n = len(values)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        means.append(sample.mean())
    means = np.asarray(means)
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return lower, upper


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = os.path.join("data", "radar_sequences.npz")
    dataset = RadarSequenceDataset(npz_path)

    x0, _, _ = dataset[0]
    T, C, H, W = x0.shape
    num_regions = H * W

    ckpt = torch.load(os.path.join("checkpoints", "forecaster.pt"), map_location=device)
    model = ExtremeEventForecaster(
        in_channels=C,
        hidden_dim=32,
        num_gcn_layers=2,
        gcn_hidden_dim=32,
        num_regions=num_regions,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    A_hat = build_grid_adjacency(H, W, device)

    temp_channel_idx = 0  # adjust if temperature is stored in another channel

    probs_baseline = []
    probs_15 = []
    probs_20 = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, yg, yr = dataset[i]
            x = x.unsqueeze(0).to(device)  # [1, T, C, H, W]

            x_base = x
            x_15 = apply_warming_scenario(x, temp_channel_idx, 1.5)
            x_20 = apply_warming_scenario(x, temp_channel_idx, 2.0)

            g_base, _ = model(x_base, A_hat)
            g_15, _ = model(x_15, A_hat)
            g_20, _ = model(x_20, A_hat)

            p_base = torch.sigmoid(g_base.squeeze(-1)).item()
            p_15 = torch.sigmoid(g_15.squeeze(-1)).item()
            p_20 = torch.sigmoid(g_20.squeeze(-1)).item()

            probs_baseline.append(p_base)
            probs_15.append(p_15)
            probs_20.append(p_20)

    mean_base = np.mean(probs_baseline)
    mean_15 = np.mean(probs_15)
    mean_20 = np.mean(probs_20)

    ci_base = bootstrap_ci(probs_baseline)
    ci_15 = bootstrap_ci(probs_15)
    ci_20 = bootstrap_ci(probs_20)

    print("Baseline mean P(E):", mean_base, "CI:", ci_base)
    print("+1.5C   mean P(E):", mean_15, "CI:", ci_15)
    print("+2.0C   mean P(E):", mean_20, "CI:", ci_20)

    np.savez(
        os.path.join("results", "scenario_probs.npz"),
        probs_baseline=probs_baseline,
        probs_15=probs_15,
        probs_20=probs_20,
        mean_base=mean_base,
        mean_15=mean_15,
        mean_20=mean_20,
        ci_base=np.array(ci_base),
        ci_15=np.array(ci_15),
        ci_20=np.array(ci_20),
    )


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
