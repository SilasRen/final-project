"""
Feature engineering utilities for time series data
"""
import numpy as np
import pandas as pd

def make_supervised(df: pd.DataFrame, target_col: str, seq_len: int):
    """
    Convert time series data into supervised learning format
    
    Creates sequences of length seq_len as input features (X) and 
    the next value as target (y) for sequence-to-one prediction.
    
    Args:
        df: DataFrame containing time series data
        target_col: Name of the target column
        seq_len: Length of input sequences
    
    Returns:
        X: Array of shape (n_samples, seq_len) - input sequences
        y: Array of shape (n_samples,) - target values
    
    Example:
        If seq_len=3 and data is [1, 2, 3, 4, 5]:
        X = [[1, 2, 3], [2, 3, 4]]
        y = [4, 5]
    """
    X, y = [], []
    vals = df[target_col].values
    for i in range(seq_len, len(vals)):
        X.append(vals[i-seq_len:i])
        y.append(vals[i])
    return np.array(X), np.array(y)
