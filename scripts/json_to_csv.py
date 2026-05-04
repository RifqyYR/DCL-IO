"""
Convert Metrics History JSON to CSV
======================================
Converts metrics_history.json from fine-tuning or pretraining
into a CSV table for easy viewing and reporting.

Usage:
    # Fine-tuning metrics
    python scripts/json_to_csv.py --history output/finetune/v3/metrics_history.json

    # Pretraining metrics
    python scripts/json_to_csv.py --history output/pretrain/full/metrics_history.json

    # Custom output path
    python scripts/json_to_csv.py --history output/finetune/v3/metrics_history.json --output results.csv
"""

import argparse
import csv
import json
import os
import sys


def detect_mode(history):
    """Auto-detect whether history is from pretraining or fine-tuning."""
    if 'loss_total' in history and 'loss_single' in history:
        return 'pretrain'
    elif 'train_loss' in history:
        return 'finetune'
    return 'unknown'


def convert_finetune(history, output_path):
    """Convert fine-tuning metrics_history.json to CSV."""
    epochs = history['epoch']
    val_epoch_set = set(history.get('val_epoch', []))

    # Build a lookup for val metrics by epoch
    val_lookup = {}
    for i, ve in enumerate(history.get('val_epoch', [])):
        val_lookup[ve] = {
            'val_mAP': history['val_mAP'][i] if i < len(history['val_mAP']) else '',
            'val_AP50': history['val_AP50'][i] if i < len(history['val_AP50']) else '',
            'val_AP75': history['val_AP75'][i] if i < len(history['val_AP75']) else '',
        }

    headers = [
        'epoch',
        'train_loss',
        'loss_classifier',
        'loss_box_reg',
        'loss_objectness',
        'loss_rpn_box_reg',
        'learning_rate',
        'val_mAP',
        'val_AP50',
        'val_AP75',
    ]

    rows = []
    for i, ep in enumerate(epochs):
        row = {
            'epoch': ep,
            'train_loss': f"{history['train_loss'][i]:.6f}",
            'loss_classifier': f"{history['train_loss_classifier'][i]:.6f}",
            'loss_box_reg': f"{history['train_loss_box_reg'][i]:.6f}",
            'loss_objectness': f"{history['train_loss_objectness'][i]:.6f}",
            'loss_rpn_box_reg': f"{history['train_loss_rpn_box_reg'][i]:.6f}",
            'learning_rate': f"{history['lr'][i]:.8f}",
            'val_mAP': '',
            'val_AP50': '',
            'val_AP75': '',
        }

        if ep in val_lookup:
            v = val_lookup[ep]
            row['val_mAP'] = f"{v['val_mAP']:.4f}" if v['val_mAP'] != '' else ''
            row['val_AP50'] = f"{v['val_AP50']:.4f}" if v['val_AP50'] != '' else ''
            row['val_AP75'] = f"{v['val_AP75']:.4f}" if v['val_AP75'] != '' else ''

        rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return headers, rows


def convert_pretrain(history, output_path):
    """Convert pretraining metrics_history.json to CSV."""
    headers = ['epoch', 'loss_total', 'loss_single', 'loss_dense', 'learning_rate']

    rows = []
    for i, ep in enumerate(history['epoch']):
        rows.append({
            'epoch': ep,
            'loss_total': f"{history['loss_total'][i]:.6f}",
            'loss_single': f"{history['loss_single'][i]:.6f}",
            'loss_dense': f"{history['loss_dense'][i]:.6f}",
            'learning_rate': f"{history['lr'][i]:.8f}",
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return headers, rows


def print_table(headers, rows, max_rows=None):
    """Print a formatted table to console."""
    display_rows = rows if max_rows is None else rows[:max_rows]

    # Column widths
    widths = {h: len(h) for h in headers}
    for row in display_rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ''))))

    # Header
    header_line = ' | '.join(h.center(widths[h]) for h in headers)
    separator = '-+-'.join('-' * widths[h] for h in headers)

    print(f"\n{header_line}")
    print(separator)
    for row in display_rows:
        line = ' | '.join(str(row.get(h, '')).rjust(widths[h]) for h in headers)
        print(line)

    if max_rows and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows)")

    print(f"\nTotal: {len(rows)} epochs\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert metrics_history.json to CSV"
    )
    parser.add_argument('--history', type=str, required=True,
                        help='Path to metrics_history.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: same dir as input)')
    parser.add_argument('--print_table', action='store_true', default=True,
                        help='Print table to console')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Max rows to print (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.history, 'r') as f:
        history = json.load(f)

    mode = detect_mode(history)

    # Default output path
    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.history)[0] + '.csv'

    if mode == 'finetune':
        headers, rows = convert_finetune(history, output_path)
    elif mode == 'pretrain':
        headers, rows = convert_pretrain(history, output_path)
    else:
        print("Error: Could not detect metrics type.")
        sys.exit(1)

    print(f"Mode: {mode}")
    print(f"Saved CSV to: {output_path}")

    if args.print_table:
        print_table(headers, rows, max_rows=args.max_rows)


if __name__ == '__main__':
    main()
