import numpy as np
from utils.metrics import sigmoid


def baseline_persistence_logits(X):
    precip_last = X[:, -1, 2, :, :]  # assumes channels [temp, hum, precip]
    score = precip_last.mean(axis=(1, 2))
    z = (score - score.mean()) / (score.std() + 1e-8)
    return z.astype(np.float32)


def pooled_feature_matrix(X):
    B, T, C, H, W = X.shape
    feats = []
    for c in range(C):
        v = X[:, :, c, :, :].reshape(B, T, -1)
        feats.append(v.mean(axis=(1, 2)))
        feats.append(v.std(axis=(1, 2)))
        feats.append(v.max(axis=(1, 2)))
    return np.stack(feats, axis=1).astype(np.float32)


def train_logistic_regression(X, y, lr=0.5, l2=1e-3, steps=800):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    N, D = X.shape
    w = np.zeros(D, dtype=np.float32)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = (X.T @ (p - y)) / N + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return w, float(b)


def logistic_logits(X, w, b):
    return (X @ w + b).astype(np.float32)
