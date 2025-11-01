import argparse, yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    fig_dir = Path(cfg['paths']['fig_dir']); fig_dir.mkdir(parents=True, exist_ok=True)
    # 这里先留空位，后续接入你训练好的预测结果文件
    x = np.linspace(0, 10, 200)
    y = np.sin(x)
    plt.figure(figsize=(6,3)); plt.plot(x,y); plt.title('Demo Evaluate')
    out = fig_dir/'evaluate_demo.png'
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f'[SAVE] {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    main(args.config)
