"""
Modified DenseCL Pretraining Script
======================================
Self-supervised pretraining using Modified DenseCL on unlabeled 
dental panoramic images.

Supports ablation study via --ablation flag:
  baseline  = vanilla DenseCL
  mod1      = + Soft Lesion-Aware Weighting
  mod2      = + Asymmetric Augmentation
  mod3      = + Hard Negative Mining
  full      = all 3 modifications

Usage:
    python scripts/pretrain.py --data_dir data/unlabeled/filtered --ablation full
    python scripts/pretrain.py --data_dir data/unlabeled/filtered --ablation baseline --epochs 200
"""

import argparse
import json
import os
import sys
import time
import random

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    Config, get_config, get_baseline_config, get_mod1_config,
    get_mod2_config, get_mod3_config, get_full_config
)
from data.dataset_pretraining import DentalPanoramicPretrainDataset, pretrain_collate_fn
from models.modified_densecl import ModifiedDenseCL
from utils.hard_negative_mining import HardNegativeMiner


def parse_args():
    parser = argparse.ArgumentParser(description="Modified DenseCL Pretraining")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing filtered unlabeled images')
    parser.add_argument('--output_dir', type=str, default='output/pretrain',
                        help='Output directory for checkpoints and logs')
    
    # Ablation
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['baseline', 'mod1', 'mod2', 'mod3', 'full'],
                        help='Ablation configuration')
    
    # Training params (override config)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(cfg):
    """Build Modified DenseCL model from config."""
    model = ModifiedDenseCL(
        backbone_cfg=dict(
            depth=50, pretrained=False, return_stages=[4]
        ),
        neck_cfg=dict(
            in_channels=cfg.densecl.backbone_out_channels,
            hid_channels=cfg.densecl.neck_hid_channels,
            out_channels=cfg.densecl.neck_out_channels,
            num_grid=cfg.densecl.num_grid,
        ),
        head_cfg=dict(temperature=cfg.densecl.temperature),
        queue_len=cfg.densecl.queue_len,
        feat_dim=cfg.densecl.feat_dim,
        momentum=cfg.densecl.momentum,
        loss_lambda=cfg.densecl.loss_lambda,
        use_lesion_weighting=cfg.ablation.use_lesion_weighting,
        lesion_alpha=cfg.lesion_weighting.alpha,
        lesion_gaussian_sigma=cfg.lesion_weighting.gaussian_sigma,
        use_hard_negative_mining=cfg.ablation.use_hard_negative_mining,
        hard_neg_warmup_epochs=cfg.hard_negative.warmup_epochs,
        hard_neg_top_k=cfg.hard_negative.top_k,
        hard_neg_loss_weight=cfg.hard_negative.hard_neg_loss_weight,
    )
    
    return model


def build_dataset(cfg, data_dir):
    """Build pretraining dataset from config."""
    dataset = DentalPanoramicPretrainDataset(
        image_dir=data_dir,
        global_size=cfg.augmentation.global_crop_size,
        local_size=cfg.augmentation.local_crop_size,
        use_asymmetric_aug=cfg.ablation.use_asymmetric_aug,
        use_lesion_weighting=cfg.ablation.use_lesion_weighting,
        lesion_crop_bias=cfg.augmentation.lesion_crop_bias,
        context_padding_factor=cfg.augmentation.context_padding_factor,
        augmentation_config=cfg.augmentation,
        pseudo_lesion_config=cfg.pseudo_lesion,
    )
    return dataset


def build_optimizer(cfg, model):
    """Build optimizer."""
    if cfg.pretrain.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.pretrain.lr,
            weight_decay=cfg.pretrain.weight_decay,
            momentum=cfg.pretrain.sgd_momentum,
        )
    elif cfg.pretrain.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.pretrain.lr,
            weight_decay=cfg.pretrain.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.pretrain.optimizer}")
    
    return optimizer


def build_scheduler(cfg, optimizer, total_steps):
    """Build learning rate scheduler."""
    if cfg.pretrain.scheduler == 'cosine':
        # Cosine annealing with warmup
        warmup_steps = cfg.pretrain.warmup_epochs  # Will be multiplied by steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(cfg.pretrain.min_lr / cfg.pretrain.lr,
                      0.5 * (1 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, 
                    epoch, cfg, writer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_single_loss = 0.0
    total_dense_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.pretrain.epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        view1 = batch['view1'].to(device)
        view2 = batch['view2'].to(device)
        weight_maps = batch['weight_map']
        
        if weight_maps is not None:
            weight_maps = weight_maps.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if cfg.pretrain.use_amp and device != 'cpu':
            with autocast():
                losses = model(view1, view2, mode='train', weight_maps=weight_maps)
                loss = losses['loss_total']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model(view1, view2, mode='train', weight_maps=weight_maps)
            loss = losses['loss_total']
            
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track losses
        total_loss += loss.item()
        total_single_loss += losses['loss_contra_single'].item()
        total_dense_loss += losses['loss_contra_dense'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'single': f"{losses['loss_contra_single'].item():.4f}",
            'dense': f"{losses['loss_contra_dense'].item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
        })
        
        # TensorBoard logging
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % cfg.pretrain.log_interval == 0:
            writer.add_scalar('train/loss_total', loss.item(), global_step)
            writer.add_scalar('train/loss_single', 
                            losses['loss_contra_single'].item(), global_step)
            writer.add_scalar('train/loss_dense', 
                            losses['loss_contra_dense'].item(), global_step)
            writer.add_scalar('train/lr', 
                            optimizer.param_groups[0]['lr'], global_step)
    
    # Epoch averages
    avg_loss = total_loss / max(num_batches, 1)
    avg_single = total_single_loss / max(num_batches, 1)
    avg_dense = total_dense_loss / max(num_batches, 1)
    
    writer.add_scalar('epoch/loss_total', avg_loss, epoch)
    writer.add_scalar('epoch/loss_single', avg_single, epoch)
    writer.add_scalar('epoch/loss_dense', avg_dense, epoch)
    
    return avg_loss, avg_single, avg_dense


def main():
    args = parse_args()
    
    # Load ablation config
    config_builders = {
        'baseline': get_baseline_config,
        'mod1': get_mod1_config,
        'mod2': get_mod2_config,
        'mod3': get_mod3_config,
        'full': get_full_config,
    }
    cfg = config_builders[args.ablation]()
    
    # Override config from CLI args
    if args.epochs is not None:
        cfg.pretrain.epochs = args.epochs
    if args.batch_size is not None:
        cfg.pretrain.batch_size = args.batch_size
    if args.lr is not None:
        cfg.pretrain.lr = args.lr
    if args.num_workers is not None:
        cfg.pretrain.num_workers = args.num_workers
    if args.resume is not None:
        cfg.pretrain.resume_checkpoint = args.resume
    
    cfg.device = args.device
    cfg.seed = args.seed
    
    # Setup output directories
    output_dir = os.path.join(args.output_dir, args.ablation)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    tb_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    # Metrics history file
    history_path = os.path.join(output_dir, 'metrics_history.json')
    
    # Set seed
    set_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print ablation config
    print(f"\n{'='*60}")
    print(f"Modified DenseCL Pretraining — Ablation: {args.ablation}")
    print(f"{'='*60}")
    print(f"  Lesion-Aware Weighting: {cfg.ablation.use_lesion_weighting}")
    print(f"  Asymmetric Augmentation: {cfg.ablation.use_asymmetric_aug}")
    print(f"  Hard Negative Mining: {cfg.ablation.use_hard_negative_mining}")
    print(f"  Epochs: {cfg.pretrain.epochs}")
    print(f"  Batch Size: {cfg.pretrain.batch_size}")
    print(f"  Learning Rate: {cfg.pretrain.lr}")
    print(f"  Global Crop: {cfg.augmentation.global_crop_size}")
    print(f"  Local Crop: {cfg.augmentation.local_crop_size}")
    print(f"{'='*60}\n")
    
    # Build dataset
    dataset = build_dataset(cfg, args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=True,
        num_workers=cfg.pretrain.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pretrain_collate_fn,
    )
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    total_steps = cfg.pretrain.epochs * len(dataloader)
    scheduler = build_scheduler(cfg, optimizer, total_steps)
    
    # Mixed precision scaler
    scaler = GradScaler() if cfg.pretrain.use_amp else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    if cfg.pretrain.resume_checkpoint is not None:
        print(f"Resuming from {cfg.pretrain.resume_checkpoint}")
        checkpoint = torch.load(cfg.pretrain.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch}")
    
    # ========================================================================
    # Load or initialize metrics history
    # ========================================================================
    if os.path.exists(history_path) and cfg.pretrain.resume_checkpoint is not None:
        with open(history_path, 'r') as f:
            history = json.load(f)
        for key in history:
            if isinstance(history[key], list):
                history[key] = history[key][:start_epoch]
        print(f"  Loaded metrics history ({len(history.get('epoch', []))} epochs)")
    else:
        history = {
            'epoch': [],
            'loss_total': [],
            'loss_single': [],
            'loss_dense': [],
            'lr': [],
        }
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print(f"\nStarting training from epoch {start_epoch}...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, cfg.pretrain.epochs):
        # Update model epoch (for hard negative warmup)
        model.current_epoch = epoch
        
        # Train one epoch
        epoch_start = time.time()
        avg_loss, avg_single, avg_dense = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            epoch, cfg, writer, device
        )
        epoch_time = time.time() - epoch_start
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{cfg.pretrain.epochs} — "
              f"Loss: {avg_loss:.4f} — Time: {epoch_time:.1f}s")
        
        # --- Record metrics ---
        history['epoch'].append(epoch + 1)
        history['loss_total'].append(avg_loss)
        history['loss_single'].append(avg_single)
        history['loss_dense'].append(avg_dense)
        history['lr'].append(current_lr)
        
        # --- Save metrics history after every epoch ---
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save checkpoint
        if (epoch + 1) % cfg.pretrain.checkpoint_interval == 0 or \
           epoch == cfg.pretrain.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'ablation': args.ablation,
                    'epochs': cfg.pretrain.epochs,
                    'batch_size': cfg.pretrain.batch_size,
                    'lr': cfg.pretrain.lr,
                },
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, best_path)
    
    # ========================================================================
    # Save final backbone weights for downstream
    # ========================================================================
    backbone_weights = model.extract_backbone_weights()
    backbone_path = os.path.join(output_dir, 'backbone_weights.pth')
    torch.save(backbone_weights, backbone_path)
    print(f"\nBackbone weights saved to: {backbone_path}")
    
    writer.close()
    print(f"\nPretraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")
    print(f"TensorBoard logs in: {tb_dir}")
    print(f"Metrics history saved to: {history_path}")


if __name__ == '__main__':
    main()
