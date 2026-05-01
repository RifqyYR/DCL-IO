"""
Faster R-CNN Fine-tuning Script
==================================
Fine-tune Faster R-CNN with FPN on labeled dental panoramic images
using pretrained DenseCL backbone for IO detection.

Usage:
    python scripts/finetune.py \
        --image_dir data/labeled/images \
        --annotation_file data/labeled/annotations.json \
        --backbone_weights output/pretrain/full/backbone_weights.pth \
        --output_dir output/finetune/full
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from data.dataset_finetuning import build_detection_datasets
from models.faster_rcnn import build_faster_rcnn
from utils.metrics import evaluate_detection, print_eval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN Fine-tuning")
    
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing labeled images')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='COCO-format annotation JSON file')
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help='Pretrained backbone weights from DenseCL')
    parser.add_argument('--output_dir', type=str, default='output/finetune',
                        help='Output directory')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1,
                        help='LR multiplier for backbone (lower = less overfit)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step'],
                        help='LR scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='LR warmup epochs')
    parser.add_argument('--lr_step_size', type=int, default=15,
                        help='Step size for StepLR (only if scheduler=step)')
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum LR for cosine scheduler')
    
    # EMA (Exponential Moving Average)
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model for evaluation/saving')
    parser.add_argument('--ema_decay', type=float, default=0.9998,
                        help='EMA decay factor')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (0 = disabled)')
    
    # Validation frequency
    parser.add_argument('--eval_interval', type=int, default=3,
                        help='Evaluate every N epochs')
    
    # Model params
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (background + IO)')
    parser.add_argument('--trainable_layers', type=int, default=3,
                        help='Number of trainable backbone layers')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def detection_collate_fn(batch):
    """Collate function for detection datasets."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs,
                    ema_model=None, ema_decay=0.9998):
    """Train Faster R-CNN for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip batches with no annotations
        valid = [len(t['boxes']) > 0 for t in targets]
        if not any(valid):
            continue
        
        optimizer.zero_grad()
        
        # Forward pass (returns loss dict in training mode)
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # --- Update EMA per batch (critical: must be per-step, not per-epoch) ---
        if ema_model is not None:
            _update_ema(model, ema_model, ema_decay)
        
        # Track losses
        total_loss += losses.item()
        total_loss_classifier += loss_dict.get('loss_classifier', 
                                                torch.tensor(0.0)).item()
        total_loss_box_reg += loss_dict.get('loss_box_reg', 
                                             torch.tensor(0.0)).item()
        total_loss_objectness += loss_dict.get('loss_objectness', 
                                                torch.tensor(0.0)).item()
        total_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 
                                                 torch.tensor(0.0)).item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'cls': f"{loss_dict.get('loss_classifier', torch.tensor(0.0)).item():.4f}",
            'box': f"{loss_dict.get('loss_box_reg', torch.tensor(0.0)).item():.4f}",
        })
    
    avg_loss = total_loss / max(num_batches, 1)
    return {
        'total': avg_loss,
        'classifier': total_loss_classifier / max(num_batches, 1),
        'box_reg': total_loss_box_reg / max(num_batches, 1),
        'objectness': total_loss_objectness / max(num_batches, 1),
        'rpn_box_reg': total_loss_rpn_box_reg / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate Faster R-CNN on validation set."""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        
        # Forward pass (returns detections in eval mode)
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            pred = {
                'boxes': output['boxes'].cpu(),
                'scores': output['scores'].cpu(),
                'labels': output['labels'].cpu(),
            }
            gt = {
                'boxes': target['boxes'],
                'labels': target['labels'],
            }
            all_predictions.append(pred)
            all_ground_truths.append(gt)
    
    # Compute metrics
    results = evaluate_detection(all_predictions, all_ground_truths)
    return results


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Output dirs
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Metrics history file
    history_path = os.path.join(args.output_dir, 'metrics_history.json')
    
    print(f"\n{'='*60}")
    print(f"Faster R-CNN Fine-tuning for IO Detection")
    print(f"{'='*60}")
    print(f"  Backbone weights: {args.backbone_weights}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Classes: {args.num_classes}")
    print(f"{'='*60}\n")
    
    # Build datasets
    train_dataset, val_dataset = build_detection_datasets(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )
    
    # Build model
    model = build_faster_rcnn(
        num_classes=args.num_classes,
        pretrained_backbone_path=args.backbone_weights,
        trainable_backbone_layers=args.trainable_layers,
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ========================================================================
    # Differential LR: lower LR for backbone, higher for head/FPN
    # ========================================================================
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'body' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor,
         'name': 'backbone'},
        {'params': head_params, 'lr': args.lr, 'name': 'head'},
    ]
    
    optimizer = torch.optim.SGD(
        param_groups, momentum=0.9, weight_decay=args.weight_decay
    )
    
    print(f"Backbone LR: {args.lr * args.backbone_lr_factor:.6f}, "
          f"Head LR: {args.lr:.6f}")
    
    # ========================================================================
    # Learning rate scheduler
    # ========================================================================
    if args.scheduler == 'cosine':
        # Cosine annealing with linear warmup
        warmup_epochs = args.warmup_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
                return max(args.min_lr / args.lr,
                          0.5 * (1 + np.cos(np.pi * progress)))
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    
    # ========================================================================
    # EMA (Exponential Moving Average) model
    # ========================================================================
    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"EMA enabled (decay={args.ema_decay})")
    
    # ========================================================================
    # Resume from checkpoint
    # ========================================================================
    start_epoch = 0
    best_map = 0.0
    epochs_no_improve = 0
    
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'ema_state_dict' in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_map = checkpoint.get('best_map', checkpoint.get('mAP', 0.0))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"  Resumed from epoch {start_epoch}, best mAP so far: {best_map:.4f}")
    
    # ========================================================================
    # Load or initialize metrics history
    # ========================================================================
    if os.path.exists(history_path) and args.resume is not None:
        with open(history_path, 'r') as f:
            history = json.load(f)
        # Trim history to start_epoch in case of partial resume
        for key in history:
            if isinstance(history[key], list):
                history[key] = history[key][:start_epoch]
        print(f"  Loaded metrics history ({len(history.get('epoch', []))} epochs)")
    else:
        history = {
            'epoch': [],
            'train_loss': [],
            'train_loss_classifier': [],
            'train_loss_box_reg': [],
            'train_loss_objectness': [],
            'train_loss_rpn_box_reg': [],
            'lr': [],
            'val_mAP': [],
            'val_AP50': [],
            'val_AP75': [],
            'val_epoch': [],
        }
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # Train
        epoch_start = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs,
            ema_model=ema_model, ema_decay=args.ema_decay
        )
        lr_scheduler.step()
        epoch_time = time.time() - epoch_start
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} \u2014 "
              f"Loss: {train_losses['total']:.4f} \u2014 "
              f"Time: {epoch_time:.1f}s \u2014 "
              f"LR: {current_lr:.6f}")
        
        # --- Record training metrics ---
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_losses['total'])
        history['train_loss_classifier'].append(train_losses['classifier'])
        history['train_loss_box_reg'].append(train_losses['box_reg'])
        history['train_loss_objectness'].append(train_losses['objectness'])
        history['train_loss_rpn_box_reg'].append(train_losses['rpn_box_reg'])
        history['lr'].append(current_lr)
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            # Use EMA model for evaluation if available
            eval_model = ema_model if ema_model is not None else model
            eval_results = evaluate(eval_model, val_loader, device)
            print_eval_results(eval_results)
            
            # --- Record validation metrics ---
            history['val_epoch'].append(epoch + 1)
            history['val_mAP'].append(float(eval_results['mAP']))
            history['val_AP50'].append(float(eval_results.get('AP50', 0.0)))
            history['val_AP75'].append(float(eval_results.get('AP75', 0.0)))
            
            # Save best model (EMA weights if available)
            current_map = eval_results['mAP']
            if current_map > best_map:
                best_map = current_map
                epochs_no_improve = 0
                best_path = os.path.join(ckpt_dir, 'best.pth')
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': (ema_model if ema_model is not None else model).state_dict(),
                    'mAP': current_map,
                    'eval_results': eval_results,
                }
                torch.save(save_dict, best_path)
                print(f"  \u2605 New best mAP: {best_map:.4f}, saved to {best_path}")
            else:
                epochs_no_improve += args.eval_interval
            
            # --- Early stopping ---
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\n  Early stopping triggered! No improvement for "
                      f"{epochs_no_improve} epochs (patience={args.patience}).")
                break
        
        # --- Save metrics history after every epoch ---
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_map': best_map,
                'epochs_no_improve': epochs_no_improve,
            }
            if ema_model is not None:
                save_dict['ema_state_dict'] = ema_model.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
    
    print(f"\nFine-tuning complete! Best mAP: {best_map:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")
    print(f"Metrics history saved to: {history_path}")


@torch.no_grad()
def _update_ema(model, ema_model, decay):
    """Update EMA model parameters.
    
    EMA averages model weights over training, producing smoother
    and more generalizable weights that reduce overfitting.
    
    ema_param = decay * ema_param + (1 - decay) * model_param
    """
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
    
    # Also update buffers (batch norm stats etc.)
    for ema_buf, model_buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.data.copy_(model_buf.data)


if __name__ == '__main__':
    main()

