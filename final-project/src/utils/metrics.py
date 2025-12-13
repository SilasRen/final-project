import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss_from_logits(logits, y, eps=1e-12):
    p = sigmoid(logits)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def confusion_counts(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def precision_recall_f1(tp, tn, fp, fn, eps=1e-12):
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2.0 * prec * rec / (prec + rec + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return float(acc), float(prec), float(rec), float(f1)


def roc_auc_score(y_true, y_score):
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())

    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / pos
    fpr = fps / neg

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    return float(np.trapz(tpr, fpr))


def compute_binary_metrics(y_true, logits, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.float32)
    logits = np.asarray(logits).astype(np.float32)
    probs = sigmoid(logits)
    y_pred = (probs >= threshold).astype(np.int64)

    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    acc, prec, rec, f1 = precision_recall_f1(tp, tn, fp, fn)
    auc = roc_auc_score(y_true, probs)
    bce = bce_loss_from_logits(logits, y_true)

    return {
        "bce": bce,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
