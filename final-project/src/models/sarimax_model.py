from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pickle
from pathlib import Path

def fit_sarimax(series, order=(1,1,1), seasonal_order=(1,1,1,12), verbose=True):
    if verbose:
        print(f'[FITTING] SARIMAX with order={order}, seasonal_order={seasonal_order}')
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    if verbose:
        print(f'[FITTED] SARIMAX AIC={res.aic:.2f}')
    return res

def save_sarimax(fitted, path):
    """Save SARIMAX model"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(fitted, f)
    print(f'[SAVE] SARIMAX model saved to {path}')

def load_sarimax(path):
    """Load SARIMAX model"""
    with open(path, 'rb') as f:
        fitted = pickle.load(f)
    print(f'[LOAD] SARIMAX model loaded from {path}')
    return fitted

def forecast_sarimax(fitted, steps):
    fc = fitted.forecast(steps=steps)
    return np.asarray(fc)
