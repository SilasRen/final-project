import pandas as pd
import numpy as np
from pathlib import Path

def load_or_make_demo_csv(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return pd.read_csv(p, parse_dates=['Date'])
    idx = pd.date_range('1900-01-01', periods=1500, freq='MS')
    rng = np.random.default_rng(42)
    trend = np.linspace(-0.2, 1.2, len(idx))
    season = 0.2*np.sin(2*np.pi*np.arange(len(idx))/12)
    noise = rng.normal(0, 0.1, len(idx))
    df = pd.DataFrame({'Date': idx, 'TempAnomaly': trend + season + noise})
    df.to_csv(p, index=False)
    return df

def moving_average(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    df = df.sort_values('Date').copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df['MA'] = df[col].rolling(window=window, center=True, min_periods=window).mean()
    return df.dropna(subset=['MA'])
