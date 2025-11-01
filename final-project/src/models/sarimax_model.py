from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def fit_sarimax(series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_sarimax(fitted, steps):
    fc = fitted.forecast(steps=steps)
    return np.asarray(fc)
