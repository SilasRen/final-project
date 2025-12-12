import os
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None
    print("[Warning] PyTorch not installed. Using demo data.")


FEATURE_NAMES = [
    "Temperature",
    "Precipitation",
    "Wind Speed",
    "Humidity",
    "Pressure",
    "Solar Radiation",
]

MAX_PLOTS = 30

TORCH_TENSOR_PATH = os.path.join("results", "extreme_events_tensor.pt")
NUMPY_ARRAY_PATH = os.path.join("results", "extreme_events_array.npy")
OUTPUT_DIR = os.path.join("results", "radar_charts")


def load_features() -> np.ndarray:
    if torch is not None and os.path.exists(TORCH_TENSOR_PATH):
        print(f"[Info] Loading PyTorch tensor from {TORCH_TENSOR_PATH} ...")
        obj = torch.load(TORCH_TENSOR_PATH, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict) and "features" in obj:
            arr = obj["features"]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
        else:
            raise ValueError("Invalid tensor format in .pt file.")

        arr = np.asarray(arr, dtype=float)
        print(f"[Info] Loaded tensor, shape = {arr.shape}")
        return arr

    if os.path.exists(NUMPY_ARRAY_PATH):
        print(f"[Info] Loading NumPy array from {NUMPY_ARRAY_PATH} ...")
        arr = np.load(NUMPY_ARRAY_PATH)
        arr = np.asarray(arr, dtype=float)
        print(f"[Info] Loaded array, shape = {arr.shape}")
        return arr

    print("[Warning] No data file found. Using demo random data.")
    num_events = 50
    num_features = len(FEATURE_NAMES)
    rng = np.random.default_rng(seed=42)
    arr = rng.random((num_events, num_features))
    print(f"[Info] Demo data generated, shape = {arr.shape}")
    return arr


def plot_radar(ax, values, feature_names, title=""):
    values = np.asarray(values, dtype=float)
    num_vars = len(feature_names)

    if values.shape[0] != num_vars:
        raise ValueError("Dimension mismatch between values and feature names.")

    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    values = np.concatenate([values, values[:1]])

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=10, pad=10)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Info] Radar charts will be saved to: {OUTPUT_DIR}")

    features = load_features()
    num_events, num_features = features.shape

    if num_features != len(FEATURE_NAMES):
        print("[Warning] Feature dimension mismatch. Adjusting automatically.")
        if num_features > len(FEATURE_NAMES):
            features = features[:, : len(FEATURE_NAMES)]
        else:
            pad = np.zeros((num_events, len(FEATURE_NAMES) - num_features))
            features = np.concatenate([features, pad], axis=1)
        num_features = len(FEATURE_NAMES)

    num_plots = min(num_events, MAX_PLOTS)
    print(f"[Info] Total events: {num_events}. Plotting first {num_plots}.")

    for idx in range(num_plots):
        values = features[idx]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        title = f"Extreme Event #{idx + 1}"
        plot_radar(ax, values, FEATURE_NAMES, title)

        out_path = os.path.join(OUTPUT_DIR, f"radar_event_{idx + 1:03d}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        if (idx + 1) % 5 == 0 or (idx + 1) == num_plots:
            print(f"[Info] Generated {idx + 1}/{num_plots} radar charts")

    print("[Done] Completed radar chart generation.")


if __name__ == "__main__":
    main()
