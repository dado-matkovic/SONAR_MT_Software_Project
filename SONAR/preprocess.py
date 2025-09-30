#!C:/Users/nas/Desktop/STUD/ss 25/SONAR_PROJECT/venv/Scripts/python.exe

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import argparse
import yaml

from utils import load_datasets

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SonarDataPreprocessor:
    def __init__(self, 
                 max_length: int = 512,
                 min_length: int = 3):
        
        self.max_length = max_length
        self.min_length = min_length

        self.setup_tokenizer()
        self.stats = defaultdict(int)
    
    def setup_tokenizer(self):
        #try:
        #    self.tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
        #    logger.info("Loaded SONAR tokenizer")
        #except Exception as e:
        #    logger.warning(f"Could not load SONAR tokenizer: {e}")
        #    logger.info("Using NLLB tokenizer as fallback")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", additional_special_tokens=["deu_Latn", "fra_Latn", "spa_Latn", "eng_Latn"])
    
    def clean_text(self, text: str) -> str:
    
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_validate(self, tokenizer, text: str, lang: str, max_length: int = 512) -> bool:
        try:
            # Set language for NLLB tokenizer
            if hasattr(tokenizer, 'src_lang'):
                tokenizer.src_lang = lang
                
            tokens = tokenizer(text)

            if max_length is not None and len(tokens) > max_length:
                text = text[:max_length]         
                
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed for {lang}: {e}")
            return False
    
    def process_single_sample(self, sample: Dict, flip=False) -> Optional[Dict]:
        self.stats['total_samples'] += 1
        translation = sample['translation']
        laser_score= sample['laser_score']

        langs = list(translation.keys())
        if len(langs) != 2:
            self.stats['filtered_languages_no_pair'] += 1
            return None
            
        lang1, lang2 = langs
        text1 = self.clean_text(translation[lang1])
        text2 = self.clean_text(translation[lang2])
     
        text1 = self.tokenize_and_validate(self.tokenizer, text1, lang1, self.max_length)
        text2 = self.tokenize_and_validate(self.tokenizer, text2, lang2, self.max_length)

        if not flip:
            return {
                lang1: text1,
                lang2: text2,
                'laser_score': laser_score
            }
        elif flip:
            return {
                lang2: text2,
                lang1: text1,
                'laser_score': laser_score
            }
    
    def print_statistics(self):
        logger.info("=== Processing Statistics ===")
        
        # General stats
        logger.info(f"Total samples processed: {self.stats['total_samples']}")
        logger.info(f"Filtered too short: {self.stats['filtered_languages_no_pair']}")

    def process_datasets(self, dataset: Dataset) -> List[Dict]:
        logger.info(f"Processing dataset {dataset}")
        
        processed_samples = []
        dataset_size = dataset.num_rows
        
        for i in range(dataset_size):
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i}/{dataset_size} samples...")
            
            sample = dataset[i]
            processed_sample = self.process_single_sample(sample, flip=False)
            processed_sample_flipped = self.process_single_sample(sample, flip=True)

            if processed_sample:
                processed_samples.append(processed_sample)

            if processed_sample_flipped:
                processed_samples.append(processed_sample_flipped)
        
        logger.info(f"Processed {len(processed_samples)} valid samples from {dataset_size}")
        return processed_samples 

def create_train_dev_split(pair_data: List[Dict], dev_ratio=0.05) -> Tuple[List[Dict], List[Dict]]:
    data_copy = pair_data.copy()
    random.shuffle(data_copy)
    dev_size = int(len(data_copy) * dev_ratio)
    dev_data = data_copy[:dev_size]
    train_data = data_copy[dev_size:]
    
    return train_data, dev_data

def flatten_language_pairs(data: List[Dict]) -> Dict[str, List[Dict]]:
    
    language_pairs = {}
    
    for sample in data:
        langs = list(sample.keys())
        if len(langs) != 3:
            logger.warning(f"Sample has {len(langs)} languages, skipping: {langs}")
            continue
        langs.remove('laser_score')
        lang1_so, lang2_so = sorted(langs)
        lang1, lang2 = langs
        pair_key = f"{lang1_so}-{lang2_so}"
        flattened_sample = {
            'src_lang': lang1,
            'tgt_lang': lang2,
            'src_input_ids': sample[lang1]['input_ids'],
            'src_attention_mask': sample[lang1]['attention_mask'],
            'tgt_input_ids': sample[lang2]['input_ids'],
            'tgt_attention_mask': sample[lang2]['attention_mask'],
            'laser_score':sample['laser_score']
        }
        if pair_key not in language_pairs:
            language_pairs[pair_key] = []
        language_pairs[pair_key].append(flattened_sample)
    
    # Log statistics
    logger.info("Language pair statistics:")
    for pair_key, samples in language_pairs.items():
        logger.info(f"{pair_key}: {len(samples)} samples")
    
    return language_pairs

def save_processed_pairs(processed_pairs: Dict[str, List], output_dir: str):
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_data, dev_data = create_train_dev_split(processed_pairs)
        # Flatten both train and dev
        train_pairs = flatten_language_pairs(train_data)
        dev_pairs = flatten_language_pairs(dev_data)
        
        # Save each language pair
        for pair_key in train_pairs.keys():
            logger.info(f"Saving {pair_key}...")
            
            train_dataset = Dataset.from_list(train_pairs[pair_key])
            dev_dataset = Dataset.from_list(dev_pairs.get(pair_key, []))
            
            dataset_dict = DatasetDict({
                'train': train_dataset
            })
            if len(dev_dataset) > 0:
                dataset_dict['validation'] = dev_dataset
            
            # Save
            save_path = output_path / f"{pair_key.replace('-', '_')}_tokenized"
            dataset_dict.save_to_disk(str(save_path))
            
            logger.info(f"✅ Saved {pair_key}: train={len(train_dataset)}, dev={len(dev_dataset)}")

def main():
    
    parser = argparse.ArgumentParser(description="Preprocess SONAR training data")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml")
    
    args = parser.parse_args()
    
    # Load config from YAML
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {args.config}")
    else:
        logger.info(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Load datasets
    logger.info(f"Loading {len(config['dataset_paths'])} datasets...")
    datasets = load_datasets(config['dataset_paths'], 'train')
    
    logger.info(f"datasets loaded and look like this {datasets}")

    # Initialize preprocessor
    preprocessor_config = {
        'max_length': config['max_length'],
        'min_length': config['min_length']
    }
    preprocessor = SonarDataPreprocessor(**preprocessor_config)
    all_processed_samples = []
    
    for i, dataset in enumerate(datasets):
        logger.info(f"\n Processing dataset {i+1}/{len(datasets)}...")
        
        processed_samples = preprocessor.process_datasets(dataset )
        
        all_processed_samples.extend(processed_samples)
        logger.info(f" Added {len(processed_samples)} samples")
    
    logger.info(f"\n Total processed samples: {len(all_processed_samples)}")
    
    # Print final statistics
    preprocessor.print_statistics()
    
    logger.info(" Preprocessing completed!")
    logger.info(f" saving processed data to: {config['output']['output_dir']}")

    save_processed_pairs(all_processed_samples, config['output']['output_dir'])

if __name__ == "__main__":
    main()

    