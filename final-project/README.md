# AI for Global Warming Modeling & Short-Term Extremes

A machine learning project for climate prediction using time series models.

## Project Overview

- **Long-term prediction**: Berkeley Earth temperature anomalies using SARIMAX + LSTM
- **Short-term prediction**: NOAA/NEXRAD extreme events (extensible with ConvLSTM/GNN)

## Features

- Automatic data download from Berkeley Earth and NOAA
- Multiple model types: SARIMAX, LSTM, and hybrid SARIMAX+LSTM
- Comprehensive evaluation metrics and visualization
- Configurable via YAML configuration files

## Installation

### Windows Users

See [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for detailed Windows setup instructions.

Quick start:
1. Install Anaconda/Miniconda
2. Open Anaconda Prompt
3. Navigate to project directory
4. Run: `conda env create -f environment.yml`
5. Run: `conda activate climate`
6. Double-click `run.bat` or follow manual steps

### Linux/Mac Users

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate climate
```

2. Prepare data (downloads real data from Berkeley Earth):
```bash
make data
# or
python -m src.prepare_data --config configs/default.yaml --use-real-data
```

**Important**: The script will automatically download real data from Berkeley Earth (and NOAA GSOM if `NOAA_TOKEN` is set). 
For detailed instructions on using real data, see [RUN_GUIDE.md](RUN_GUIDE.md).

**NOAA Token**: To enable NOAA GSOM downloads you **must** set the `NOAA_TOKEN` environment variable with your NOAA CDO API token:
```bash
# Windows (Command Prompt)
setx NOAA_TOKEN "your_token_here"

# macOS/Linux
export NOAA_TOKEN="your_token_here"
```
Request a token from https://www.ncei.noaa.gov/cdo-web/token.

## Usage

### Training

Train models with default configuration:
```bash
make train
# or
python -m src.train --config configs/default.yaml
```

### Evaluation

Evaluate trained models:
```bash
make evaluate
# or
python -m src.evaluate --config configs/default.yaml
```

### Complete Pipeline

Run data preparation, training, and evaluation:
```bash
make all
```

## Project Structure

```
final-project/
├── configs/
│   └── default.yaml          # Configuration file
├── data/
│   ├── raw/                  # Raw data from Berkeley Earth/NOAA
│   └── processed/            # Processed data for training
├── outputs/
│   ├── figures/              # Generated plots and visualizations
│   └── models/               # Trained model files
├── src/
│   ├── data.py               # Data loading and processing
│   ├── features.py           # Feature engineering
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── prepare_data.py       # Data preparation script
│   └── models/
│       ├── sarimax_model.py  # SARIMAX model implementation
│       └── lstm_model.py     # LSTM model implementation
└── README.md
```

## Configuration

Edit `configs/default.yaml` to customize:
- Data paths and sources
- Model hyperparameters
- Training settings
- Output directories

## Model Types

1. **SARIMAX**: Statistical time series model with seasonal components
2. **LSTM**: Deep learning model for sequence prediction
3. **SARIMAX+LSTM**: Hybrid model combining both approaches (40% SARIMAX + 60% LSTM)

## Data Sources

- **Berkeley Earth**: Global temperature anomaly data
  - Automatically downloads from official URLs
  - Falls back to synthetic data if download fails
- **NOAA**: Climate data for extreme events (future extension)

## Outputs

After training and evaluation, you'll find:
- Model files in `outputs/models/`
- Forecast plots in `outputs/figures/`
- Evaluation metrics and visualizations

## License

This project is for educational and research purposes.
