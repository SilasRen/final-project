import os
import numpy as np


def generate_climate_system(
    T_total=400,
    H=32,
    W=32,
    storm_strength=3.0,
    temp_gradient_strength=0.8,
    humidity_decay=0.97,
    noise_level=0.1,
):
    temps = []
    hums = []
    precs = []

    temp = np.linspace(-1, 1, H).reshape(H, 1) + np.zeros((H, W))
    humidity = np.zeros((H, W), dtype=np.float32)
    precip = np.zeros((H, W), dtype=np.float32)

    cx, cy = H // 2, W // 2

    for t in range(T_total):
        temp += 0.01
        temp += np.random.randn(H, W) * 0.03

        humidity = humidity * humidity_decay + np.random.rand(H, W) * 0.2

        cx += np.random.randint(-1, 2)
        cy += np.random.randint(-1, 2)
        cx = np.clip(cx, 3, H - 4)
        cy = np.clip(cy, 3, W - 4)

        storm = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                d = np.sqrt((i - cx)**2 + (j - cy)**2)
                if d < 4:
                    storm[i, j] = storm_strength * np.exp(-(d**2) / 4)

        precip = storm + humidity * 0.5 + noise_level * np.random.rand(H, W)

        temps.append(temp.copy())
        hums.append(humidity.copy())
        precs.append(precip.copy())

    temps = np.stack(temps)
    hums = np.stack(hums)
    precs = np.stack(precs)

    X_full = np.stack([temps, hums, precs], axis=2)
    return X_full


def build_sequences(X_full, T_in=8, extreme_quantile=0.95):
    T_total, H, W, C = X_full.shape

    precip = X_full[:, :, :, 2]
    thr = np.quantile(precip, extreme_quantile)

    X_list = []
    y_global_list = []
    y_region_list = []

    for t in range(T_in, T_total):
        seq = X_full[t - T_in : t]
        target = precip[t]

        extreme_mask = (target >= thr).astype(np.float32)
        y_global = float(extreme_mask.max() > 0)
        y_region = extreme_mask.reshape(-1)

        X_list.append(seq.transpose(0, 3, 1, 2))
        y_global_list.append(y_global)
        y_region_list.append(y_region)

    X = np.stack(X_list).astype(np.float32)
    y_global = np.array(y_global_list, dtype=np.float32)
    y_region = np.stack(y_region_list).astype(np.float32)

    max_val = X.max()
    X = X / max_val

    return X, y_global, y_region, thr


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "radar_sequences.npz")

    X_full = generate_climate_system(
        T_total=400,
        H=32,
        W=32,
        storm_strength=4.0,
        noise_level=0.15,
    )

    X, y_global, y_region, thr = build_sequences(
        X_full,
        T_in=8,
        extreme_quantile=0.96,
    )

    np.savez_compressed(
        out_path,
        X=X,
        y_global=y_global,
        y_region=y_region,
        threshold=thr,
    )

    print("Saved improved climate-like dataset to:", out_path)
    print("X:", X.shape)
    print("y_global:", y_global.shape)
    print("y_region:", y_region.shape)
    print("Extreme threshold:", thr)


if __name__ == "__main__":
    main()
