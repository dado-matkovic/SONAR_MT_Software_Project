#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
from collections import defaultdict
import argparse
import yaml
from functools import partial

import bitsandbytes as bnb

from utils import load_datasets, dyn_padding_collate_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import Dataset 
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoModelForSeq2SeqLM
)
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
wandb.login()
class SONARDataset(Dataset):
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            'src_tokens': sample['src_input_ids'],
            'tgt_tokens': sample['tgt_input_ids'],
            'src_attention': sample['src_attention_mask'],
            'tgt_attention': sample['tgt_attention_mask'],
            'src_lang': sample['src_lang'],
            'tgt_lang': sample['tgt_lang'],
            'laser_score':sample['laser_score']
        }

class SonarEncoderDecoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config['device'])
        # Initialize from NLLB-1 weights
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['nllb_model_name'])
        self.nllb_model= self.nllb_model.to(self.device)
        logger.info(f"Loaded NLLB model: {config['model']['nllb_model_name']}")
        
        # Encoder: Maps text to universal embedding space
        self.encoder = self._build_encoder()
        
        # Decoder: Maps embeddings back to text
        self.decoder = self._build_decoder()
        
        # Projection layer to fixed-size embeddings
        self.model_dim = self.encoder.embed_tokens.embedding_dim
        logger.info(f"Model dimension from NLLB: {self.model_dim}")
        self.sentence_embedding_projection = nn.Linear(self.model_dim, config['model']['embedding_dim'])

        # Initialize new components only (keep NLLB weights frozen initially)
        self._init_sonar_components()
        
    def gradient_checkpointing_enable_model(self):
        """Enable gradient checkpointing on the underlying model"""
        if hasattr(self.nllb_model, 'gradient_checkpointing_enable'):
            self.nllb_model.gradient_checkpointing_enable()
        
    def _init_sonar_components(self):
        """Initialize only the SONAR-specific components we added"""
        # Initialize sentence embedding projection (this is the key SONAR component)
        nn.init.xavier_uniform_(self.sentence_embedding_projection.weight)
        if self.sentence_embedding_projection.bias is not None:
            nn.init.zeros_(self.sentence_embedding_projection.bias)
        
        # Initialize language embeddings  
        #nn.init.normal_(self.language_embeddings.weight, mean=0, std=0.02)
        
        logger.info("Initialized SONAR-specific components (keeping NLLB-200 weights)")
        logger.info("This matches the SONAR paper initialization procedure")

    def _build_encoder(self):
        #try nllb1 encoder
        """try:
            config = SonarTextEncoderConfig(
                model_dim=1024,
                vocab_info=VocabularyInfo(
                    size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
                ),
                learned_pos=False,
                no_scale_embedding=False,
                emb_dropout_p=0.1,
                attention_dropout_p=0.1,
                activation_dropout_p=0.1,
                max_seq_len=512,
                pooling="mean",
                no_token_positional_embeddings=False,
                layernorm_embedding=False,
                activation_fn="ReLU",
                normalize_before=False,
                num_encoder_layers=24,
                num_decoder_layers=24,
                num_encoder_attn_heads=16,
                num_decoder_attn_heads=16,
                ffn_inner_dim=1024*8,
                _from_fairseq=True
            )
            encoder= SonarTextEncoderFactory(config).create_model()        

        except Exception as e:
            logger.warning(f"Could not load Sonar encoder: {e}")
            encoder= None"""
        encoder= self.nllb_model.model.encoder
        return encoder
        
        
    
    def _build_decoder(self):
        """config = SonarTextDecoderConfig(
                model_dim=1024,
                max_seq_len=512,
                vocab_info=VocabularyInfo(
                    size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
                ),
                learned_pos=False,
                no_scale_embedding=False,
                emb_dropout_p=0.1,
                attention_dropout_p=0.1,
                activation_dropout_p=0.1,
                no_token_positional_embeddings=False,
                layernorm_embedding=False,
                activation_fn="ReLU",
                normalize_before=True,
                num_encoder_layers=24,
                num_decoder_layers=24,
                num_encoder_attn_heads=16,
                num_decoder_attn_heads=16,
                ffn_inner_dim=1024 * 8)"""
        #decoder= SonarTextDecoderFactory(config).create_model() 
        decoder= self.nllb_model.model.decoder
        return decoder
    
    def _mean_pool(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1).float()
        summed = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        
        return summed / lengths.clamp(min=1.0)
    
    def encode(self, input_ids, attention_mask):
        
        encoder_outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # pooling to collapse each sentence into one fixed sized vec
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hid_dim]
        pooled = self._mean_pool(hidden_states, attention_mask)  # [batch, hid_dim]
        
        # fixed-size sentence embedding
        sentence_embeddings = self.sentence_embedding_projection(pooled)  # [batch, embed_dim]
        
        # normalise
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        
        return sentence_embeddings
        
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, noisy_input_ids_src):

        # send everything on same device
        src_input_ids = src_input_ids.to(self.device)
        src_attention_mask = src_attention_mask.to(self.device)
        tgt_input_ids = tgt_input_ids.to(self.device)
        tgt_attention_mask = tgt_attention_mask.to(self.device)
        noisy_input_ids_src = noisy_input_ids_src.to(self.device)
        # 1. Encode SOURCE sentence to universal embedding
        src_embeddings =self.encode(src_input_ids, src_attention_mask)
        
        # 2. Encode TARGET sentence to universal embedding  
        tgt_embeddings=self.encode(tgt_input_ids, tgt_attention_mask)

        noisy_src_embeddings= self.encoder(noisy_input_ids_src, src_attention_mask)
        
        # Full encoder output [batch, seq_len, hid_dim]
        src_encoder_outputs = self.encoder(src_input_ids, src_attention_mask)
        
        decoder_outputs = self.decoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            encoder_hidden_states=src_encoder_outputs.last_hidden_state,
            encoder_attention_mask=src_attention_mask
        )
 

        noisy_decoder_outputs = self.decoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            encoder_hidden_states=noisy_src_embeddings.last_hidden_state,
            encoder_attention_mask=src_attention_mask
        )
        return {
            'src_embeddings': src_embeddings,     
            'tgt_embeddings': tgt_embeddings,   
            'decoder_outputs': decoder_outputs,
            'noisy_decoder_outputs' : noisy_decoder_outputs
        }

class SonarTrainer:
    
    def __init__(self, config):
        self.config = config

        self.accelerator = Accelerator (gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            mixed_precision='fp16' if config['training']['fp16'] else 'no',
            cpu=False,
            device_placement=True,
            split_batches=True 
            )
        self.device = torch.device(self.config['device'])

        # Set seeds
        set_seed(config['seed'])
        accelerate_set_seed(config['seed'])
        
        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project=config['logging']['wandb_project'],
                name=config['logging']['wandb_run_name'],
                config=config
            )
    
    def setup_data(self):
        logger.info("Setting up train and dev")
        
        # Load SONAR tokenizer
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer'], additional_special_tokens=self.config['data']['languages'])
        self.vocab_size = self.tokenizer.vocab_size
        #load dataset /remove title /combine
        logger.info(f"Loading train and test datasets")
        self.train_data =load_datasets(self.config['data']['train_data_paths'], 'None')
        self.dev_data = load_datasets(self.config['data']['dev_data_paths'],'None')
        logger.info(f"datasets loaded and look like this {self.train_data} and {self.dev_data}")

        # Create training and dev dataset
        self.train_dataset = SONARDataset(self.train_data, self.config)
        self.dev_dataset = SONARDataset(self.dev_data, self.config)
        
        #collate = lambda batch: dyn_padding_collate_fn(batch, self.tokenizer)
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=partial(dyn_padding_collate_fn, tokenizer=self.tokenizer))
        

        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=partial(dyn_padding_collate_fn, tokenizer=self.tokenizer))
        
    
    def setup_model(self):
        logger.info("Set up model")
    
        self.model = SonarEncoderDecoder(self.config)
        self.model = self.model.to(self.device)
        logger.info(f" about to checkpoint grad")
        self.model.gradient_checkpointing_enable_model()

        #optim
        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            betas=(0.9, 0.999),
            weight_decay=self.config['training']['weight_decay']
        )
        #scheduler
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        #accelerator
        self.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.scheduler)
        
    def corrupt(x, noise_std, config):
        # x: tensor in [B,C,H,W]
        noise_std= config['training']['noise_std']
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        return x_noisy.clamp(0.0, 1.0)
        
    def compute_loss(self, batch, model_output):
        # different losses
        embeddings_src = model_output['src_embeddings']
        embeddings_tgt = model_output['tgt_embeddings']
        decoder_outputs = model_output['decoder_outputs']
        noisy_decoder_outputs = model_output['noisy_decoder_outputs']
        losses = {}
        

        # 1) Translation loss aka (cross-entropy)
        tgt_tokens = batch['input_ids_tgt']
        #logits 
        decoder_hidden_dim = decoder_outputs.last_hidden_state.size(-1) 
        device = decoder_outputs.last_hidden_state.device 
        lm = nn.Linear(decoder_hidden_dim, self.vocab_size).to(device)
        decoder_logits = lm(decoder_outputs.last_hidden_state)
        tran_loss = F.cross_entropy(
            decoder_logits.view(-1, decoder_logits.size(-1)),
            tgt_tokens.view(-1),
            ignore_index=1 
        )

        losses['translation'] = tran_loss * self.config['training']['mt']
        
        # 2. Embedding similarity loss (cosine similarity) MSE
        # Encourage similar meanings to have similar embeddings
        batch_size = self.config['training']['batch_size']
        if batch_size > 1:
            # Compute pairwise cosine similarities
            sim = F.cosine_similarity(
                embeddings_src.unsqueeze(1),
                embeddings_tgt.unsqueeze(0),
                dim=2
            )
            
            # Create target similarities based on LASER scores
            laser_scores = batch['laser_score']
            laser_sim = torch.outer(laser_scores, laser_scores)
            
            # MSE loss between similarities
            sim_loss = F.mse_loss(sim, laser_sim)
            losses['similarity'] = sim_loss * self.config['training']['mse']
        # 3. Reconstruction loss
        # denoising auto-encoding: noise embeddings 
        rec_loss = F.mse_loss(noisy_decoder_outputs.last_hidden_state, decoder_outputs.last_hidden_state)
        losses['reconstruction'] = rec_loss * self.config['training']['dae']
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def train_step(self, batch):
        self.model.train()

        with self.accelerator.accumulate(self.model):
            # Forward pass
            model_output = self.model(batch['input_ids_src'], batch['attention_src'], batch['input_ids_tgt'], batch['attention_tgt'], batch['noisy_input_ids_src'])
            
            # Compute loss
            losses = self.compute_loss(batch, model_output)
            
            # Backward pass
            self.accelerator.backward(losses['total'])
            
            # Clip gradients
            if self.config['training']['max_grad_norm'] > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return losses
    
    def dev_step(self, batch):
        with self.accelerator.accumulate(self.model):
            # Forward pass
            model_output = self.model(batch['input_ids_src'], batch['attention_src'], batch['input_ids_tgt'], batch['attention_tgt'], batch['noisy_input_ids_src'])
            
            # Compute loss
            losses = self.compute_loss(batch, model_output)

        return losses
    
    def save_checkpoint(self, step):
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(self.config['logging']['output_dir']) / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save({
                'step': step,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config
            }, checkpoint_dir / 'pytorch_model.bin')
            
            # Save config
            with open(checkpoint_dir / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved checkpoint at step {step}")
    
    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            self.model.train()
            epoch_losses = defaultdict(float)
            num_batches = 0
            
            for batch in self.train_loader:
                # Train
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                num_batches += 1
                
            # Logging
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            log_dict = {
                'epoch': epoch,
                'learning_rate': self.scheduler.get_last_lr()[0],
                **{f'train/{k}': v for k, v in avg_losses.items()}
            }
                
            wandb.log(log_dict, step=epoch)
            logger.info(f"epoch {epoch}: {avg_losses}")
            
            # Save checkpoint
            if epoch % self.config['training']['save_steps'] == 0:
                self.save_checkpoint(epoch)

            #eval 
            logger.info(f"Running evaluation at step {epoch}")
            self.model.eval()
            val_losses = defaultdict(float)
            num_val_batches = 0
            
            with torch.no_grad():
                for dev_batch  in self.dev_loader:
                    losses_dev = self.dev_step(dev_batch)
                    
                    # Accumulate validation losses
                    for key, value in losses_dev.items():
                        val_losses[key] += value.item() if torch.is_tensor(value) else value
                    num_val_batches += 1
    
                # Average the losses
                avg_val_losses = {k: v/num_val_batches for k, v in val_losses.items()}
                    
                # Log validation results
                val_loss_str = " | ".join([f"val_{k}: {v:.4f}" for k, v in avg_val_losses.items()])
                logger.info(f"epoch {epoch} Validation: {val_loss_str}")
                
                # Log to wandb if available
                if hasattr(self, 'use_wandb') and self.use_wandb:
                    wandb_dict = {f"val/{k}": v for k, v in avg_val_losses.items()}
                    wandb_dict["epoch"] = epoch
                    wandb.log(wandb_dict)
                    
                # Save best model based on validation loss
                if hasattr(self, 'best_val_loss'):
                    current_val_loss = avg_val_losses.get('total_loss', float('inf'))
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.save_checkpoint(epoch, is_best=True)
                        logger.info(f" New best model! Val loss: {current_val_loss:.4f}")
                else:
                    self.best_val_loss = avg_val_losses.get('total_loss', float('inf'))
            
            # End of epoch
            logger.info(f"Epoch {epoch + 1} completed. Average losses: {avg_losses}")
            torch.cuda.empty_cache()

        logger.info("Training completed!")
        wandb.finish()

def main():
    #import gc
    #torch.cuda.empty_cache()
    #gc.collect()
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    #torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="Train SONAR")
    parser.add_argument("--config", type=str, default="/app/config/train_config.yaml")
    
    args = parser.parse_args()
    #logger.info(f"path is {os.getcwd()}")
    #config_dir=(Path("SONAR_PROJECT/config"))
    #logger.info(f" config dir exists: {config_dir.exists()}")
    # Load config from YAML
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {args.config}")
    else:
        logger.info(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # init
    trainer = SonarTrainer(config)

    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()