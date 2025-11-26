"""
Evaluation script for climate prediction models
Loads trained models and evaluates them on test data
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_or_make_demo_csv, moving_average
from src.features import make_supervised
from src.models.sarimax_model import load_sarimax, forecast_sarimax
from src.models.lstm_model import load_lstm, predict_lstm

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any NaN or inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'correlation': np.nan
        }
    
    # RMSE (Root Mean Squared Error)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    
    # MAE (Mean Absolute Error)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Correlation coefficient
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'correlation': correlation
    }

def plot_predictions(df_ma, train_split_idx, y_true, y_pred, metrics, model_type, fig_dir):
    """
    Create comprehensive evaluation plots
    
    Args:
        df_ma: DataFrame with moving average data
        train_split_idx: Index where train/test split occurs
        y_true: True values
        y_pred: Predicted values
        metrics: Dictionary of evaluation metrics
        model_type: Type of model used
        fig_dir: Directory to save figures
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Time series forecast
    plt.figure(figsize=(14, 6))
    plt.plot(df_ma['Date'], df_ma['y'], label='30y Moving Average', alpha=0.7, linewidth=1.5, color='blue')
    
    # Plot test period
    test_dates = df_ma['Date'].iloc[train_split_idx:train_split_idx+len(y_pred)]
    plt.plot(test_dates, y_true, label='True Values', alpha=0.8, linewidth=2, color='green')
    plt.plot(test_dates, y_pred, label=f'Predictions ({model_type})', alpha=0.8, linewidth=2, color='red', linestyle='--')
    
    plt.axvline(df_ma['Date'].iloc[train_split_idx], color='orange', linestyle='--', 
                alpha=0.7, linewidth=2, label='Train/Test Split')
    plt.legend(fontsize=10)
    plt.title(f'Forecast vs Actual (RMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}, R²={metrics["r2"]:.4f})', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'evaluation_forecast.png', dpi=150, bbox_inches='tight')
    print(f'[SAVE] Saved forecast plot to {fig_dir / "evaluation_forecast.png"}')
    plt.close()
    
    # Plot 2: Scatter plot (predicted vs actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values (°C)', fontsize=11)
    plt.ylabel('Predicted Values (°C)', fontsize=11)
    plt.title(f'Predicted vs Actual (R²={metrics["r2"]:.4f}, Corr={metrics["correlation"]:.4f})', 
              fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'evaluation_scatter.png', dpi=150, bbox_inches='tight')
    print(f'[SAVE] Saved scatter plot to {fig_dir / "evaluation_scatter.png"}')
    plt.close()
    
    # Plot 3: Residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values (°C)', fontsize=11)
    plt.ylabel('Residuals (°C)', fontsize=11)
    plt.title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (°C)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'Residuals Distribution (Mean={np.mean(residuals):.4f})', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / 'evaluation_residuals.png', dpi=150, bbox_inches='tight')
    print(f'[SAVE] Saved residuals plot to {fig_dir / "evaluation_residuals.png"}')
    plt.close()

def main(cfg_path):
    """
    Main evaluation function
    
    Args:
        cfg_path: Path to configuration file
    """
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    
    print('=' * 60)
    print('Model Evaluation')
    print('=' * 60)
    
    # Load data
    print('\n[LOAD] Loading data...')
    df = load_or_make_demo_csv(cfg['data']['be_path'], use_real_data=True)
    df_ma = moving_average(df, cfg['data']['target_col'], cfg['features']['ma_window_months'])
    df_ma = df_ma.rename(columns={'MA': 'y'})[['Date', 'y']].reset_index(drop=True)
    print(f'[DATA] Loaded {len(df_ma)} samples')
    
    # Split data (same as training)
    split = int(len(df_ma) * 0.8)
    train_series = df_ma['y'][:split]
    test_series = df_ma['y'][split:]
    print(f'[SPLIT] Train: {len(train_series)}, Test: {len(test_series)}')
    
    model_dir = Path(cfg['paths']['model_dir'])
    model_type = cfg['model']['type']
    preds = None
    
    # Load and evaluate SARIMAX
    if model_type in ['sarimax', 'sarimax_lstm']:
        sarimax_path = model_dir / 'sarimax_model.pkl'
        if sarimax_path.exists():
            print('\n[EVAL] Loading SARIMAX model...')
            sarimax_fit = load_sarimax(str(sarimax_path))
            preds = forecast_sarimax(sarimax_fit, steps=len(test_series))
            print(f'[EVAL] SARIMAX predictions: {len(preds)} steps')
        else:
            print(f'[WARN] SARIMAX model not found at {sarimax_path}')
    
    # Load and evaluate LSTM
    if model_type in ['lstm', 'sarimax_lstm']:
        lstm_path = model_dir / 'lstm_model.pth'
        if lstm_path.exists():
            print('\n[EVAL] Loading LSTM model...')
            Xte, _ = make_supervised(df_ma[['y']].iloc[split-cfg['model']['lstm']['seq_len']:], 
                                     'y', cfg['model']['lstm']['seq_len'])
            lstm = load_lstm(str(lstm_path), 
                           hidden_size=cfg['model']['lstm']['hidden_size'],
                           num_layers=cfg['model']['lstm']['num_layers'])
            lstm_pred = predict_lstm(lstm, Xte)
            print(f'[EVAL] LSTM predictions: {len(lstm_pred)} steps')
            
            if preds is None:
                preds = lstm_pred
            else:
                # Combine predictions: 40% SARIMAX + 60% LSTM
                preds = 0.4 * preds[:len(lstm_pred)] + 0.6 * np.asarray(lstm_pred)
        else:
            print(f'[WARN] LSTM model not found at {lstm_path}')
    
    if preds is None:
        print('[ERROR] No models found. Please run training first.')
        return
    
    # Calculate metrics
    y_true = test_series.values[-len(preds):]
    metrics = calculate_metrics(y_true, preds)
    
    # Print results
    print('\n' + '=' * 60)
    print('Evaluation Results')
    print('=' * 60)
    print(f'Model Type: {model_type}')
    print(f'RMSE:  {metrics["rmse"]:.6f} °C')
    print(f'MAE:   {metrics["mae"]:.6f} °C')
    print(f'MAPE:  {metrics["mape"]:.2f} %')
    print(f'R²:    {metrics["r2"]:.6f}')
    print(f'Correlation: {metrics["correlation"]:.6f}')
    print('=' * 60)
    
    # Create plots
    print('\n[PLOT] Generating evaluation plots...')
    plot_predictions(df_ma, split, y_true, preds, metrics, model_type, cfg['paths']['fig_dir'])
    
    # Save metrics to file
    metrics_path = Path(cfg['paths']['fig_dir']) / 'evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write('Evaluation Metrics\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model Type: {model_type}\n')
        f.write(f'RMSE:  {metrics["rmse"]:.6f} °C\n')
        f.write(f'MAE:   {metrics["mae"]:.6f} °C\n')
        f.write(f'MAPE:  {metrics["mape"]:.2f} %\n')
        f.write(f'R²:    {metrics["r2"]:.6f}\n')
        f.write(f'Correlation: {metrics["correlation"]:.6f}\n')
    print(f'[SAVE] Saved metrics to {metrics_path}')
    
    print('\n[COMPLETE] Evaluation finished!')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate trained climate prediction models')
    ap.add_argument('--config', default='configs/default.yaml', help='Configuration file path')
    args = ap.parse_args()
    main(args.config)
