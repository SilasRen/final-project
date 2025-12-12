# src/extreme_events.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_PATH = OUTPUT_DIR / "evaluation_extreme.txt"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_monthly_anomaly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    date_candidates = [c for c in df.columns if c.lower() in ["date", "time", "month", "yearmonth", "year_mon"]]

    if not date_candidates:
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                date_candidates.append(c)
        if not date_candidates:
            for c in df.columns:
                cl = c.lower()
                if "date" in cl or "time" in cl or "month" in cl:
                    date_candidates.append(c)

    if not date_candidates:
        raise ValueError(f"Cannot infer date column in {path}, columns={df.columns}")

    date_col = date_candidates[0]

    numeric_cols = [c for c in df.columns if c != date_col and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError(f"Cannot infer anomaly column in {path}, columns={df.columns}")

    anom_col = numeric_cols[0]

    df = df[[date_col, anom_col]].rename(columns={date_col: "date", anom_col: "temp_anomaly"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df



def add_extreme_labels(df: pd.DataFrame, upper_q: float = 0.95, lower_q: float = 0.05):
    up = df["temp_anomaly"].quantile(upper_q)
    low = df["temp_anomaly"].quantile(lower_q)

    def label_fn(x):
        if x >= up:
            return 1
        elif x <= low:
            return -1
        else:
            return 0

    df = df.copy()
    df["extreme_label_3class"] = df["temp_anomaly"].apply(label_fn)
    df["is_extreme"] = (df["extreme_label_3class"] != 0).astype(int)
    return df


def build_lag_features(df: pd.DataFrame, max_lag: int = 12):
    df = df.copy()
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["temp_anomaly"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def train_extreme_classifier(df: pd.DataFrame, max_lag: int = 12, train_ratio: float = 0.8):
    df_lag = build_lag_features(df, max_lag=max_lag)

    feature_cols = [f"lag_{i}" for i in range(1, max_lag + 1)]
    X = df_lag[feature_cols].values
    y = df_lag["is_extreme"].values

    n = len(df_lag)
    split_idx = int(n * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight=weight_dict
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    with open(METRIC_PATH, "w", encoding="utf-8") as f:
        f.write("=== Extreme Month Classification (is_extreme) ===\n\n")
        f.write(report + "\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n")

    df_lag = df_lag.copy()
    df_lag["pred_is_extreme"] = 0
    df_lag.loc[df_lag.index[split_idx:], "pred_is_extreme"] = y_pred

    return df_lag, split_idx


def plot_extreme_detection(df_lag: pd.DataFrame, split_idx: int,
                           upper_q: float, lower_q: float, fig_path: Path):
    up = df_lag["temp_anomaly"].quantile(upper_q)
    low = df_lag["temp_anomaly"].quantile(lower_q)

    plt.figure(figsize=(13, 5))
    plt.plot(df_lag["date"], df_lag["temp_anomaly"], label="Monthly Anomaly", alpha=0.6)

    test_mask = df_lag.index >= split_idx

    true_ext_mask = test_mask & (df_lag["is_extreme"] == 1)
    plt.scatter(df_lag.loc[true_ext_mask, "date"],
                df_lag.loc[true_ext_mask, "temp_anomaly"],
                marker="o", s=40, edgecolor="red", facecolor="none",
                label="True Extreme (test)")

    pred_ext_mask = test_mask & (df_lag["pred_is_extreme"] == 1)
    plt.scatter(df_lag.loc[pred_ext_mask, "date"],
                df_lag.loc[pred_ext_mask, "temp_anomaly"],
                marker="x", s=50, color="red",
                label="Predicted Extreme (test)")

    plt.axhline(up, color="orange", linestyle="--")
    plt.axhline(low, color="blue", linestyle="--")

    plt.title("Extreme Climate Month Detection (Prediction vs True)")
    plt.xlabel("Date")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main():
    data_path = DATA_PROCESSED / "noaa_gsom_monthly.csv"
    df = load_monthly_anomaly(data_path)

    upper_q = 0.95
    lower_q = 0.05

    df_labeled = add_extreme_labels(df, upper_q=upper_q, lower_q=lower_q)
    df_model, split_idx = train_extreme_classifier(df_labeled, max_lag=12, train_ratio=0.8)

    fig_path = FIG_DIR / "extreme_detection.png"
    plot_extreme_detection(df_model, split_idx,
                           upper_q=upper_q, lower_q=lower_q,
                           fig_path=fig_path)


if __name__ == "__main__":
    main()
