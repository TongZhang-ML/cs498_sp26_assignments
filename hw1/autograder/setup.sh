#!/usr/bin/env bash
set -e

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib
pip install gradescope-utils>=0.2.7
pip install func_timeout
