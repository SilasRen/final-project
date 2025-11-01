#!/usr/bin/env bash
set -e
conda env create -f environment.yml || true
conda activate climate
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
