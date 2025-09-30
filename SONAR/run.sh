#!/bin/bash
# Shell script to run the Python file
echo "Starting the Python script..."
export WANDB_API_KEY="04f2ddd157e0df2f41741cf9dfa33b93e498daeb"
export WANDB_DIR="/app/wandb"
mkdir -p "$WANDB_DIR"
echo "current dict: $(pwd)"
echo "contents are: $(ls -l /app)"
python3 /app/train.py
echo "Python script completed."