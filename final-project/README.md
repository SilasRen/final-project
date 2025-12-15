# AI for Climate Trend Modeling and Short-Term Extreme Events

A machine learning project that integrates long-term global warming trend analysis
with short-term extreme climate event detection, aiming to support decision-oriented
climate risk assessment under uncertainty.

---

## Project Overview

This project implements a unified climate analytics pipeline with three components:

- Long-term climate modeling  
  Global temperature anomaly forecasting using Berkeley Earth data with
  SARIMAX, LSTM, and a hybrid SARIMAX+LSTM model.

- Short-term extreme event detection  
  Time-aligned binary classification of extreme climate events using
  spatiotemporal sequence modeling and rolling time-series cross-validation.

- Scenario-based risk analysis  
  Controlled warming perturbations (+1.5°C, +2.0°C) to evaluate changes in
  extreme-event probabilities with uncertainty quantification.

---

## Features

- Long-term temperature anomaly modeling with statistical and neural approaches
- Short-term extreme-event detection formulated as an event-alignment task
- Rolling time-series cross-validation to prevent information leakage
- Scenario simulation under alternative warming assumptions
- Bootstrap-based confidence intervals for scenario risk estimates
- Reproducible pipeline configured via YAML files

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate climate

Usage
Data Preparation
python -m src.prepare_data --config configs/default.yaml

Long-Term Model Training
python -m src.train --config configs/default.yaml

Extreme Event Cross-Validation
python -m src.train_with_cv

Evaluation
python -m src.evaluate --config configs/default.yaml

Scenario Simulation
python -m src.scenario_simulation

Project Structure
final-project/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── train_with_cv.py
│   ├── evaluate.py
│   ├── extreme_events.py
│   ├── extreme_events_radar.py
│   ├── scenario_simulation.py
│   ├── plot_scenarios.py
│   └── models/
│       ├── sarimax_model.py
│       └── lstm_model.py
└── README.md

Model Types

SARIMAX
Statistical time-series model capturing linear dynamics and seasonality.

LSTM
Neural sequence model for nonlinear temporal dependencies.

Hybrid SARIMAX + LSTM
Convex combination of statistical and neural forecasts.

Extreme Event Classifier
Binary event detection model evaluated via time-aligned predictions rather
than pointwise regression accuracy.

Data Sources

Berkeley Earth
Global monthly temperature anomaly data for long-term climate trend analysis.

Synthetic spatiotemporal sequences
Used to emulate radar-like extreme climate event patterns for short-term
detection and scenario evaluation.

Outputs

After running the pipeline, the project generates:

Forecast and diagnostic figures in outputs/figures/

Trained model files in outputs/models/

Cross-validation metrics in outputs/results/

Scenario-based extreme-event probability estimates with confidence intervals

Notes

This project emphasizes interpretability, temporal consistency, and decision relevance
rather than precise long-horizon climate prediction. The framework is extensible and can
incorporate higher-resolution observations, alternative extreme-event definitions, and
more complex climate scenarios.