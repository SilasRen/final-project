# AI for Global Warming Modeling & Short-Term Extremes

- 长期：Berkeley Earth 温度异常（SARIMAX + LSTM）
- 短期：NOAA/NEXRAD 极端事件（后续可扩展 ConvLSTM/GNN）

## 运行
conda env create -f environment.yml
conda activate climate
python -m src.train --config configs/default.yaml
