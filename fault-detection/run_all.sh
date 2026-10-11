#!/bin/sh
# Runs every task in order (from the repo root).
set -e
cd src
python load_data.py
python prepare.py
python eda.py
python train_logistic.py
python train_svm.py
python test_models.py
