"""
Data Filtering Script
======================
Run the 7-stage data filtering pipeline on raw dental panoramic images.

Usage:
    python scripts/filter_data.py --input_dir data/unlabeled/raw --output_dir data/unlabeled/filtered
    python scripts/filter_data.py --help
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, DataFilterConfig
from data.data_filter import DataFilterPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="7-stage data filtering for dental panoramic images"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw panoramic images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save filtered images')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to save filtering statistics JSON')
    
    # Override filter parameters
    parser.add_argument('--min_width', type=int, default=800)
    parser.add_argument('--min_height', type=int, default=400)
    parser.add_argument('--min_aspect_ratio', type=float, default=1.5)
    parser.add_argument('--max_aspect_ratio', type=float, default=3.5)
    parser.add_argument('--min_brightness', type=float, default=30.0)
    parser.add_argument('--max_brightness', type=float, default=220.0)
    parser.add_argument('--min_contrast', type=float, default=20.0)
    parser.add_argument('--min_sharpness', type=float, default=10.0)
    parser.add_argument('--phash_threshold', type=int, default=8)
    parser.add_argument('--no_copy', action='store_true',
                        help='Do not copy files, just print statistics')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build filter config from args
    filter_cfg = DataFilterConfig(
        min_width=args.min_width,
        min_height=args.min_height,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
        phash_threshold=args.phash_threshold,
    )
    
    # Set log file
    log_file = args.log_file
    if log_file is None:
        log_file = os.path.join(args.output_dir, 'filter_log.json')
    
    # Run pipeline
    pipeline = DataFilterPipeline(filter_cfg)
    passing_images = pipeline.run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_files=not args.no_copy,
        log_file=log_file
    )
    
    print(f"\nDone! {len(passing_images)} images passed all filters.")


if __name__ == '__main__':
    main()
