"""
Plot Training Metrics
=======================
Visualize training and validation metrics from saved metrics_history.json files.
Supports plotting pretraining and fine-tuning metrics, as well as comparing
multiple ablation runs side-by-side.

Usage:
    # Plot single fine-tuning run
    python scripts/plot_metrics.py --history output/finetune/full/metrics_history.json

    # Plot single pretraining run
    python scripts/plot_metrics.py --history output/pretrain/full/metrics_history.json --mode pretrain

    # Compare multiple ablation runs
    python scripts/plot_metrics.py \
        --compare \
        output/pretrain/baseline/metrics_history.json \
        output/pretrain/mod1/metrics_history.json \
        output/pretrain/mod2/metrics_history.json \
        output/pretrain/mod3/metrics_history.json \
        output/pretrain/full/metrics_history.json \
        --labels baseline mod1 mod2 mod3 full

    # Save plots without displaying (for server/headless)
    python scripts/plot_metrics.py --history output/finetune/full/metrics_history.json --save_dir output/plots --no_show
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_history(path):
    """Load metrics history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def detect_mode(history):
    """Auto-detect whether history is from pretraining or fine-tuning."""
    if 'loss_total' in history and 'loss_single' in history:
        return 'pretrain'
    elif 'train_loss' in history:
        return 'finetune'
    else:
        return 'unknown'


# ==========================================================================
# Pretraining plots
# ==========================================================================

def plot_pretrain(history, save_dir=None, show=True):
    """Plot pretraining metrics (losses + LR)."""
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DenseCL Pretraining Metrics', fontsize=16, fontweight='bold')
    
    # --- Total Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history['loss_total'], color='#2196F3', linewidth=1.5)
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # --- Single (Global) vs Dense Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, history['loss_single'], color='#FF9800', linewidth=1.5, 
            label='Global (Single)')
    ax.plot(epochs, history['loss_dense'], color='#4CAF50', linewidth=1.5,
            label='Dense')
    ax.set_title('Global vs Dense Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Loss components stacked ---
    ax = axes[1, 0]
    ax.fill_between(epochs, 0, history['loss_single'], alpha=0.4, 
                     color='#FF9800', label='Global')
    ax.fill_between(epochs, history['loss_single'], 
                     [s + d for s, d in zip(history['loss_single'], history['loss_dense'])],
                     alpha=0.4, color='#4CAF50', label='Dense')
    ax.set_title('Loss Composition')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Learning Rate ---
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], color='#9C27B0', linewidth=1.5)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'pretrain_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================================
# Fine-tuning plots
# ==========================================================================

def plot_finetune(history, save_dir=None, show=True):
    """Plot fine-tuning metrics (train loss breakdown + validation mAP)."""
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Faster R-CNN Fine-tuning Metrics', fontsize=16, fontweight='bold')
    
    # --- Total Training Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color='#2196F3', linewidth=1.5)
    ax.set_title('Total Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # --- Classifier Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss_classifier'], color='#FF5722', linewidth=1.5)
    ax.set_title('Classifier Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # --- Box Regression Loss ---
    ax = axes[0, 2]
    ax.plot(epochs, history['train_loss_box_reg'], color='#FF9800', linewidth=1.5)
    ax.set_title('Box Regression Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # --- Objectness + RPN Box Loss ---
    ax = axes[1, 0]
    ax.plot(epochs, history['train_loss_objectness'], color='#795548', 
            linewidth=1.5, label='Objectness')
    ax.plot(epochs, history['train_loss_rpn_box_reg'], color='#607D8B', 
            linewidth=1.5, label='RPN Box Reg')
    ax.set_title('RPN Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Validation mAP ---
    ax = axes[1, 1]
    if len(history.get('val_epoch', [])) > 0:
        val_epochs = history['val_epoch']
        ax.plot(val_epochs, history['val_mAP'], color='#4CAF50', linewidth=2,
                marker='o', markersize=5, label='mAP')
        ax.plot(val_epochs, history['val_AP50'], color='#2196F3', linewidth=1.5,
                marker='s', markersize=4, linestyle='--', label='AP50')
        ax.plot(val_epochs, history['val_AP75'], color='#F44336', linewidth=1.5,
                marker='^', markersize=4, linestyle='--', label='AP75')
        ax.legend()
        
        # Annotate best mAP
        best_idx = np.argmax(history['val_mAP'])
        best_epoch = val_epochs[best_idx]
        best_val = history['val_mAP'][best_idx]
        ax.annotate(f'Best: {best_val:.4f}\n(epoch {best_epoch})',
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 2, best_val - 0.05),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No validation data yet', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_title('Validation mAP')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.grid(True, alpha=0.3)
    
    # --- Learning Rate ---
    ax = axes[1, 2]
    ax.plot(epochs, history['lr'], color='#9C27B0', linewidth=1.5)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'finetune_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================================
# Comparison plots (ablation study)
# ==========================================================================

def plot_comparison(histories, labels, save_dir=None, show=True):
    """Compare multiple runs side-by-side (e.g., ablation study).
    
    Args:
        histories (list): List of history dicts.
        labels (list): List of labels for each run.
    """
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0',
              '#00BCD4', '#795548', '#607D8B']
    
    mode = detect_mode(histories[0])
    
    if mode == 'pretrain':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Pretraining Ablation Comparison', fontsize=16, fontweight='bold')
        
        for i, (hist, label) in enumerate(zip(histories, labels)):
            c = colors[i % len(colors)]
            epochs = hist['epoch']
            
            axes[0].plot(epochs, hist['loss_total'], color=c, linewidth=1.5, label=label)
            axes[1].plot(epochs, hist['loss_single'], color=c, linewidth=1.5, label=label)
            axes[2].plot(epochs, hist['loss_dense'], color=c, linewidth=1.5, label=label)
        
        axes[0].set_title('Total Loss')
        axes[1].set_title('Global (Single) Loss')
        axes[2].set_title('Dense Loss')
        
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    elif mode == 'finetune':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Fine-tuning Ablation Comparison', fontsize=16, fontweight='bold')
        
        for i, (hist, label) in enumerate(zip(histories, labels)):
            c = colors[i % len(colors)]
            
            axes[0].plot(hist['epoch'], hist['train_loss'], color=c, 
                        linewidth=1.5, label=label)
            
            if len(hist.get('val_epoch', [])) > 0:
                axes[1].plot(hist['val_epoch'], hist['val_mAP'], color=c,
                            linewidth=1.5, marker='o', markersize=4, label=label)
                axes[2].plot(hist['val_epoch'], hist['val_AP50'], color=c,
                            linewidth=1.5, marker='s', markersize=4, label=label)
        
        axes[0].set_title('Train Loss')
        axes[0].set_ylabel('Loss')
        axes[1].set_title('Validation mAP')
        axes[1].set_ylabel('mAP')
        axes[2].set_title('Validation AP50')
        axes[2].set_ylabel('AP50')
        
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'ablation_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================================
# Summary table
# ==========================================================================

def print_summary(history, label=""):
    """Print a summary table of the training history."""
    mode = detect_mode(history)
    
    print(f"\n{'='*60}")
    print(f"Training Summary{f' — {label}' if label else ''}")
    print(f"{'='*60}")
    print(f"  Total epochs: {len(history.get('epoch', []))}")
    
    if mode == 'pretrain':
        if len(history['loss_total']) > 0:
            print(f"  Final loss:   {history['loss_total'][-1]:.4f}")
            print(f"  Best loss:    {min(history['loss_total']):.4f} "
                  f"(epoch {history['epoch'][np.argmin(history['loss_total'])]})")
            print(f"  Final single: {history['loss_single'][-1]:.4f}")
            print(f"  Final dense:  {history['loss_dense'][-1]:.4f}")
    
    elif mode == 'finetune':
        if len(history['train_loss']) > 0:
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Best train loss:  {min(history['train_loss']):.4f} "
                  f"(epoch {history['epoch'][np.argmin(history['train_loss'])]})")
        
        if len(history.get('val_mAP', [])) > 0:
            best_idx = np.argmax(history['val_mAP'])
            print(f"\n  Best val mAP:  {history['val_mAP'][best_idx]:.4f} "
                  f"(epoch {history['val_epoch'][best_idx]})")
            print(f"  Best val AP50: {history['val_AP50'][best_idx]:.4f}")
            print(f"  Best val AP75: {history['val_AP75'][best_idx]:.4f}")
            print(f"  Last val mAP:  {history['val_mAP'][-1]:.4f} "
                  f"(epoch {history['val_epoch'][-1]})")
    
    print(f"{'='*60}")


# ==========================================================================
# Main
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from metrics_history.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run:
  python scripts/plot_metrics.py --history output/finetune/full/metrics_history.json
  
  # Pretraining:
  python scripts/plot_metrics.py --history output/pretrain/full/metrics_history.json
  
  # Compare ablation runs:
  python scripts/plot_metrics.py --compare output/pretrain/*/metrics_history.json --labels baseline mod1 mod2 mod3 full
  
  # Save without showing:
  python scripts/plot_metrics.py --history output/finetune/full/metrics_history.json --save_dir output/plots --no_show
        """
    )
    
    # Single run
    parser.add_argument('--history', type=str, default=None,
                        help='Path to a single metrics_history.json')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['pretrain', 'finetune'],
                        help='Force plot mode (auto-detected if not set)')
    
    # Comparison mode
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='Paths to multiple metrics_history.json for comparison')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each comparison run')
    
    # Output
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plot images')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots (save only)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    show = not args.no_show
    
    if args.compare:
        # --- Comparison mode ---
        histories = []
        for path in args.compare:
            histories.append(load_history(path))
        
        labels = args.labels
        if labels is None:
            # Auto-generate labels from paths
            labels = [os.path.basename(os.path.dirname(p)) for p in args.compare]
        
        if len(labels) != len(histories):
            print(f"Warning: {len(labels)} labels for {len(histories)} histories. "
                  f"Using auto-generated labels.")
            labels = [os.path.basename(os.path.dirname(p)) for p in args.compare]
        
        # Print summaries
        for hist, label in zip(histories, labels):
            print_summary(hist, label)
        
        # Plot comparison
        plot_comparison(histories, labels, save_dir=args.save_dir, show=show)
    
    elif args.history:
        # --- Single run mode ---
        history = load_history(args.history)
        mode = args.mode or detect_mode(history)
        
        print_summary(history)
        
        if mode == 'pretrain':
            plot_pretrain(history, save_dir=args.save_dir, show=show)
        elif mode == 'finetune':
            plot_finetune(history, save_dir=args.save_dir, show=show)
        else:
            print(f"Error: Could not auto-detect mode. Use --mode pretrain|finetune")
    
    else:
        print("Error: Provide --history for single run or --compare for comparison.")
        print("Run with --help for usage examples.")


if __name__ == '__main__':
    main()
