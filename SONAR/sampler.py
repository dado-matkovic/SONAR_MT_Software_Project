#!/usr/bin/env python3
"""
Simple script to randomly sample datasets
Usage: python sampler.py --config config/sample_config.yaml
"""

import os
import argparse
import yaml
import logging
from pathlib import Path
from datasets import Dataset, load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_dataset(dataset_path: str, max_samples: int = 50000, output_dir: str = "nllb_datasets/sampled_data"):

    logger.info(f"Processing: {dataset_path}")
    
    #load data
    dataset = load_from_disk(dataset_path)
    logger.info(f"Og size is: {dataset.num_rows['train']} samples")
    
    # Sample
    if dataset.num_rows['train'] > max_samples:
        dataset['train'] = dataset['train'].shuffle(seed=42).select(range(max_samples))
        logger.info(f"Sampled to: {dataset.num_rows['train']} samples")
    else:
        logger.info(f"Dataset already small!")
    
    # save
    dataset_name = Path(dataset_path).stem
    output_path = Path(output_dir) / f"{dataset_name}_sampled"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_disk(str(output_path))
    logger.info(f"Saved to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Sample datasets")
    parser.add_argument("--config", type=str, default="config/sample_config.yaml")
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config: {args.config}")
    
    # Get settings from config or defaults
    max_samples = config.get('max_samples', 50000)
    output_dir = config.get('output_dir', './nllb_datasets/sampled_data')
    dataset_paths = config.get('dataset_paths', [])
         
    logger.info(f"Sampling {len(dataset_paths)} datasets to max {max_samples:,} samples each")
    logger.info(f"Output dir: {output_dir}")
    
    sampled_paths = []
    
    for path in dataset_paths:
        try:
            sampled_path = sample_dataset(path, max_samples=max_samples, output_dir=output_dir)
            sampled_paths.append(sampled_path)
            
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
    
    logger.info(f"\nDone! Created {len(sampled_paths)} sampled datasets")
    logger.info(f"Sampled datasets saved in: {output_dir}")


if __name__ == "__main__":
    main()