SHELL := /bin/bash
setup:
\tconda env create -f environment.yml || echo env exists
train:
\tpython -m src.train --config configs/default.yaml
evaluate:
\tpython -m src.evaluate --config configs/default.yaml
