import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_path = os.path.join(base_dir, "results", "scenario_probs.npz")

    data = np.load(result_path)

    mean_base = data["mean_base"]
    mean_15 = data["mean_15"]
    mean_20 = data["mean_20"]

    ci_base = data["ci_base"]   # (lower, upper)
    ci_15 = data["ci_15"]
    ci_20 = data["ci_20"]

    # Error bars: use half-width of CI
    err_base = np.array([
        mean_base - ci_base[0],
        ci_base[1] - mean_base
    ]).reshape(2, 1)

    err_15 = np.array([
        mean_15 - ci_15[0],
        ci_15[1] - mean_15
    ]).reshape(2, 1)

    err_20 = np.array([
        mean_20 - ci_20[0],
        ci_20[1] - mean_20
    ]).reshape(2, 1)

    means = [mean_base, mean_15, mean_20]
    errors = np.hstack([err_base, err_15, err_20])

    labels = ["Baseline", "+1.5°C", "+2.0°C"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means, yerr=errors, capsize=8, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Mean Probability of Extreme Event")
    plt.title("Scenario Simulation: Extreme Event Probability Under Warming Conditions")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    out_path = os.path.join(base_dir, "results", "scenario_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Plot saved to:", out_path)


if __name__ == "__main__":
    main()
