import argparse, yaml
from pathlib import Path
import matplotlib.pyplot as plt
from src.data import load_or_make_demo_csv, moving_average
from src.features import make_supervised
from src.models.sarimax_model import fit_sarimax, forecast_sarimax
from src.models.lstm_model import train_lstm, predict_lstm

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))

    df = load_or_make_demo_csv(cfg['data']['be_path'])
    df_ma = moving_average(df, cfg['data']['target_col'], cfg['features']['ma_window_months'])
    df_ma = df_ma.rename(columns={'MA': 'y'})[['Date','y']].reset_index(drop=True)

    split = int(len(df_ma)*0.8)
    train_series = df_ma['y'][:split]
    test_series  = df_ma['y'][split:]

    preds = None
    if cfg['model']['type'] in ['sarimax','sarimax_lstm']:
        sarimax_fit = fit_sarimax(train_series,
            tuple(cfg['model']['sarimax']['order']),
            tuple(cfg['model']['sarimax']['seasonal_order']))
        preds = forecast_sarimax(sarimax_fit, steps=len(test_series))

    if cfg['model']['type'] in ['lstm','sarimax_lstm']:
        Xtr, ytr = make_supervised(df_ma[['y']].iloc[:split], 'y', cfg['model']['lstm']['seq_len'])
        Xte, yte = make_supervised(df_ma[['y']].iloc[split-cfg['model']['lstm']['seq_len']:], 'y', cfg['model']['lstm']['seq_len'])
        lstm = train_lstm(Xtr, ytr, hidden_size=cfg['model']['lstm']['hidden_size'],
                          num_layers=cfg['model']['lstm']['num_layers'],
                          lr=cfg['model']['lstm']['lr'],
                          epochs=cfg['model']['lstm']['epochs'],
                          batch_size=cfg['model']['lstm']['batch_size'])
        lstm_pred = predict_lstm(lstm, Xte)
        if preds is None: preds = lstm_pred
        else:
            import numpy as np
            preds = 0.4*preds[:len(lstm_pred)] + 0.6*np.asarray(lstm_pred)

    # 评估 & 画图
    import numpy as np
    y_true = test_series.values[-len(preds):]
    rmse = float(np.sqrt(np.mean((y_true - preds)**2)))
    print(f'[RESULT] RMSE={rmse:.4f}')

    Path(cfg['paths']['fig_dir']).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(df_ma['Date'], df_ma['y'], label='30y MA (demo)')
    plt.plot(df_ma['Date'].iloc[split:][-len(preds):], preds, label=f'Forecast ({cfg["model"]["type"]})')
    plt.legend(); plt.title(f'Forecast vs 30y MA (RMSE={rmse:.3f})')
    out = Path(cfg['paths']['fig_dir'])/'forecast_demo.png'
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f'[SAVE] {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
