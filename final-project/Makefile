SHELL := /bin/bash
setup:
	conda env create -f environment.yml || echo env exists
data:
	python -m src.prepare_data --config configs/default.yaml
train:
	python -m src.train --config configs/default.yaml
evaluate:
	python -m src.evaluate --config configs/default.yaml
all: data train evaluate
