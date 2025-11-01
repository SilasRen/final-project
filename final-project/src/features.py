import numpy as np
import pandas as pd

def make_supervised(df: pd.DataFrame, target_col: str, seq_len: int):
    X, y = [], []
    vals = df[target_col].values
    for i in range(seq_len, len(vals)):
        X.append(vals[i-seq_len:i])
        y.append(vals[i])
    return np.array(X), np.array(y)
