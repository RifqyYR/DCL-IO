"""
Extract Backbone Weights Script
==================================
Extract ResNet-50 backbone weights from a DenseCL pretraining checkpoint
for use in downstream Faster R-CNN fine-tuning.

Usage:
    python scripts/extract_backbone.py \
        --checkpoint output/pretrain/full/checkpoints/best.pth \
        --output output/pretrain/full/backbone_weights.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Backbone Weights")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DenseCL pretraining checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for backbone weights')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Extract backbone_q weights
    backbone_weights = {}
    prefixes = ['backbone_q.', 'backbone.']
    
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                backbone_weights[new_key] = value
                break
    
    if len(backbone_weights) == 0:
        print("Warning: No backbone weights found with expected prefixes.")
        print("  Trying to extract all ResNet-like keys...")
        
        # Fallback: look for ResNet layer names
        resnet_keywords = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        for key, value in state_dict.items():
            for kw in resnet_keywords:
                if kw in key:
                    # Remove any prefix before the ResNet layer name
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part in resnet_keywords:
                            new_key = '.'.join(parts[i:])
                            backbone_weights[new_key] = value
                            break
                    break
    
    print(f"Extracted {len(backbone_weights)} backbone parameters")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(backbone_weights, args.output)
    print(f"Saved backbone weights to: {args.output}")
    
    # Print parameter summary
    total_params = sum(v.numel() for v in backbone_weights.values())
    print(f"Total backbone parameters: {total_params:,}")
    
    # Print layer statistics
    layer_counts = {}
    for key in backbone_weights:
        layer = key.split('.')[0]
        if layer not in layer_counts:
            layer_counts[layer] = 0
        layer_counts[layer] += 1
    
    print("\nLayer breakdown:")
    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer}: {count} parameters")


if __name__ == '__main__':
    main()
