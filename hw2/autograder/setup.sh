#!/usr/bin/env bash
set -e

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
pip install torch==2.4.0 torchvision
pip install numpy pandas scikit-learn matplotlib
pip install gradescope-utils>=0.2.7
pip install func_timeout
pip install transformers==4.52.0
