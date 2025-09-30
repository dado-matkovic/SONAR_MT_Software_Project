#!C:/Users/nas/Desktop/STUD/ss 25/SONAR_PROJECT/venv/Scripts/python.exe
#useful methods


import logging
from typing import List
import torch 

import datasets
from datasets import Dataset, load_from_disk, concatenate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_datasets(data_paths: List[str], type: str) -> List[Dataset]:
    datasets = []
    
    for path in data_paths:
        logger.info(f"Loading dataset from {path}")
        
        dataset = load_from_disk(path)
        
        logger.info(f"Loaded dataset: {dataset.num_rows} samples")

        if type == 'None':
            datasets.append(dataset)
        else:
            datasets.append(dataset[type])
    if type == 'None':
        combined_dataset = concatenate_datasets(datasets)
        return combined_dataset
    return datasets

def dyn_padding_collate_fn(batch, tokenizer):

    input_ids_src = []
    input_ids_tgt = []
    src_attention = [] 
    tgt_attention = []
    laser_score = []
    special_tokens = get_special_tokens(tokenizer)

    for point in batch:
        input_ids_src.append(point['src_tokens'])
        input_ids_tgt.append(point['tgt_tokens'])
        src_attention.append(point['src_attention'])
        tgt_attention.append(point['tgt_attention'])
        laser_score.append(point['laser_score'])

    max_len_src = max(len(ids) for ids in input_ids_src)
    max_len_tgt = max(len(ids) for ids in input_ids_tgt)
    max_len = max(max_len_src, max_len_tgt)
    
    padded_input_ids_src = []
    attention_masks_src = []

    padded_input_ids_tgt = []
    attention_masks_tgt = []
    
    
    for ids,att in zip(input_ids_src,src_attention):
        pad_length = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = att + [0] * pad_length
        
        padded_input_ids_src.append(padded_ids)
        attention_masks_src.append(attention_mask)

    for ids,att in zip(input_ids_tgt,tgt_attention):
        pad_length = max_len - len(ids)
        padded_ids = ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = [1] * len(ids) + [0] * pad_length
        
        padded_input_ids_tgt.append(padded_ids)
        attention_masks_tgt.append(attention_mask)

    noisy_input_ids_src = random_mask(special_tokens,tokenizer.mask_token_id, torch.tensor(padded_input_ids_src), 0.15)
    return {
        'noisy_input_ids_src': noisy_input_ids_src,
        'input_ids_src': torch.tensor(padded_input_ids_src),
        'attention_src': torch.tensor(attention_masks_src),
        'input_ids_tgt': torch.tensor(padded_input_ids_tgt),
        'attention_tgt': torch.tensor(attention_masks_tgt),
        'laser_score': torch.tensor(laser_score)
    }

def get_special_tokens(tokenizer):
    special_token_ids= set([
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id
        ])
    language_token_ids= ([256042, 256057, 256047, 256121])
    special_token_ids.update(language_token_ids)
    return special_token_ids

def random_mask(special_token_ids, mask_token, input_ids, p_mask):

    torch.manual_seed(42)
    maskable = torch.ones_like(input_ids, dtype=torch.bool)
    for sp_id in special_token_ids:
        maskable &= (input_ids != sp_id)
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_positions = (rand < p_mask) & maskable 

    # Apply mask token
    input_ids[mask_positions] = mask_token

    return input_ids

