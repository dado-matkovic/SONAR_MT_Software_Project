#!/bin/bash

set -e

# Load Conda
eval "$(/nethome/dmatkovic/miniconda3/bin/conda shell.bash hook)"
conda activate py311

# Navigate to project directory
cd /nethome/dmatkovic/SONAR

# Run the translation script
python SONAR_INFERENCE_EN-FR.py
