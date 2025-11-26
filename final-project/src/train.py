"""
Training script for climate prediction models
Supports SARIMAX, LSTM, and hybrid models
"""
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.data import load_or_make_demo_csv, moving_average
from src.features import make_supervised
from src.models.sarimax_model import fit_sarimax, forecast_sarimax, save_sarimax
from src.models.lstm_model import train_lstm, predict_lstm, save_lstm

def main(cfg_path):
    """
    Main training function
    
    Args:
        cfg_path: Path to configuration YAML file
    """
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))

    print('[LOAD] Loading data...')
    df = load_or_make_demo_csv(cfg['data']['be_path'])
    df_ma = moving_average(df, cfg['data']['target_col'], cfg['features']['ma_window_months'])
    df_ma = df_ma.rename(columns={'MA': 'y'})[['Date','y']].reset_index(drop=True)
    print(f'[DATA] Loaded {len(df_ma)} samples')

    split = int(len(df_ma)*0.8)
    train_series = df_ma['y'][:split]
    test_series  = df_ma['y'][split:]
    print(f'[SPLIT] Train: {len(train_series)}, Test: {len(test_series)}')

    preds = None
    sarimax_fit = None
    lstm = None
    
    # Train SARIMAX
    if cfg['model']['type'] in ['sarimax','sarimax_lstm']:
        sarimax_fit = fit_sarimax(train_series,
            tuple(cfg['model']['sarimax']['order']),
            tuple(cfg['model']['sarimax']['seasonal_order']))
        preds = forecast_sarimax(sarimax_fit, steps=len(test_series))
        # Save SARIMAX model
        model_dir = Path(cfg['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        save_sarimax(sarimax_fit, model_dir / 'sarimax_model.pkl')

    # Train LSTM
    if cfg['model']['type'] in ['lstm','sarimax_lstm']:
        print('[TRAIN] Preparing LSTM data...')
        Xtr, ytr = make_supervised(df_ma[['y']].iloc[:split], 'y', cfg['model']['lstm']['seq_len'])
        Xte, yte = make_supervised(df_ma[['y']].iloc[split-cfg['model']['lstm']['seq_len']:], 'y', cfg['model']['lstm']['seq_len'])
        print(f'[LSTM] Train samples: {len(Xtr)}, Test samples: {len(Xte)}')
        
        lstm = train_lstm(Xtr, ytr, hidden_size=cfg['model']['lstm']['hidden_size'],
                          num_layers=cfg['model']['lstm']['num_layers'],
                          lr=cfg['model']['lstm']['lr'],
                          epochs=cfg['model']['lstm']['epochs'],
                          batch_size=cfg['model']['lstm']['batch_size'])
        lstm_pred = predict_lstm(lstm, Xte)
        
        # Save LSTM model
        model_dir = Path(cfg['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        save_lstm(lstm, model_dir / 'lstm_model.pth')
        
        if preds is None: 
            preds = lstm_pred
        else:
            # Combine predictions: 40% SARIMAX + 60% LSTM
            preds = 0.4*preds[:len(lstm_pred)] + 0.6*np.asarray(lstm_pred)

    # Evaluate & plot
    y_true = test_series.values[-len(preds):]
    rmse = float(np.sqrt(np.mean((y_true - preds)**2)))
    mae = float(np.mean(np.abs(y_true - preds)))
    mape = float(np.mean(np.abs((y_true - preds) / (y_true + 1e-8))) * 100)
    r2 = 1 - np.sum((y_true - preds)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    print(f'[RESULT] RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}')

    Path(cfg['paths']['fig_dir']).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.plot(df_ma['Date'], df_ma['y'], label='30y MA (demo)', alpha=0.7, linewidth=1)
    plt.plot(df_ma['Date'].iloc[split:][-len(preds):], preds, label=f'Forecast ({cfg["model"]["type"]})', linewidth=2)
    plt.axvline(df_ma['Date'].iloc[split], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    plt.legend()
    plt.title(f'Forecast vs 30y MA (RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f})')
    plt.xlabel('Date')
    plt.ylabel('Temperature Anomaly')
    plt.grid(True, alpha=0.3)
    out = Path(cfg['paths']['fig_dir'])/'forecast_demo.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'[SAVE] {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
