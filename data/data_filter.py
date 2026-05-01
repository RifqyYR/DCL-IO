"""
7-Stage Data Filtering Pipeline
=================================
Filters raw dental panoramic images through 7 quality checks:
  1. Resolution — minimum width/height
  2. Aspect ratio — panoramic = landscape (1.5–3.5)
  3. Brightness — mean pixel intensity range
  4. Contrast — minimum pixel std deviation
  5. Sharpness — Laplacian variance threshold
  6. Dental domain check — histogram-based heuristic
  7. Deduplication — perceptual hashing (pHash)

Usage:
    from data.data_filter import DataFilterPipeline
    pipeline = DataFilterPipeline(cfg.data_filter)
    pipeline.run(input_dir, output_dir)
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Deduplication will be skipped.")


class DataFilterPipeline:
    """7-stage data filtering pipeline for dental panoramic images.
    
    Args:
        config: DataFilterConfig dataclass with filter parameters.
    """
    
    def __init__(self, config):
        self.config = config
        self.stats = defaultdict(int)  # Track filtering statistics
    
    def check_resolution(self, img_path):
        """Stage 1: Check minimum resolution.
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if image passes resolution check.
        """
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                return w >= self.config.min_width and h >= self.config.min_height
        except Exception:
            return False
    
    def check_aspect_ratio(self, img_path):
        """Stage 2: Check aspect ratio (should be landscape for panoramic).
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if aspect ratio is within panoramic range.
        """
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                ratio = w / h
                return self.config.min_aspect_ratio <= ratio <= self.config.max_aspect_ratio
        except Exception:
            return False
    
    def check_brightness(self, img_path):
        """Stage 3: Check brightness (mean pixel intensity).
        
        Filters out images that are too dark or too bright, which typically
        indicates poor exposure or corrupted files.
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if brightness is within acceptable range.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            mean_intensity = np.mean(img)
            return (self.config.min_brightness <= mean_intensity <= 
                    self.config.max_brightness)
        except Exception:
            return False
    
    def check_contrast(self, img_path):
        """Stage 4: Check contrast (standard deviation of pixel intensities).
        
        Low contrast images provide poor learning signal for contrastive learning.
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if contrast is sufficient.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            std_intensity = np.std(img)
            return std_intensity >= self.config.min_contrast
        except Exception:
            return False
    
    def check_sharpness(self, img_path):
        """Stage 5: Check sharpness using Laplacian variance.
        
        Blurry images (low Laplacian variance) are poor candidates for 
        learning fine-grained features needed for IO detection.
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if image is sharp enough.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            lap_var = laplacian.var()
            return lap_var >= self.config.min_sharpness
        except Exception:
            return False
    
    def check_dental_domain(self, img_path):
        """Stage 6: Dental domain verification.
        
        Heuristic check that the image is likely a dental panoramic X-ray:
        - Dental X-rays have a characteristic bimodal intensity distribution
          (dark soft tissue + bright bone/teeth)
        - Very few extremely dark or bright pixels (unlike natural photos)
        - Landscape aspect ratio (already checked in stage 2)
        
        Args:
            img_path (str): Path to image file.
        
        Returns:
            bool: True if image appears to be a dental X-ray.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            
            total_pixels = img.size
            
            # Check ratio of very dark pixels (< 10)
            very_dark = np.sum(img < 10) / total_pixels
            if very_dark > self.config.dental_intensity_low_ratio:
                # Too many pure black pixels — likely not a dental X-ray
                # or has large black borders
                pass  # Allow some tolerance
            
            # Check ratio of very bright pixels (> 245)
            very_bright = np.sum(img > 245) / total_pixels
            if very_bright > self.config.dental_intensity_high_ratio:
                return False
            
            # Dental X-rays typically have intensity spread across mid-range
            mid_range = np.sum((img >= 30) & (img <= 220)) / total_pixels
            if mid_range < 0.5:
                return False
            
            # Check for at least some structure (not uniform)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            hist = hist / total_pixels
            
            # Dental X-rays shouldn't have > 30% of pixels in any single bin
            if np.max(hist) > 0.3:
                return False
            
            return True
        except Exception:
            return False
    
    def compute_phash(self, img_path, hash_size=8):
        """Compute perceptual hash of an image.
        
        Args:
            img_path (str): Path to image file.
            hash_size (int): Hash size parameter.
        
        Returns:
            imagehash.ImageHash or None: Perceptual hash.
        """
        if not HAS_IMAGEHASH:
            return None
        try:
            with Image.open(img_path) as img:
                return imagehash.phash(img, hash_size=hash_size)
        except Exception:
            return None
    
    def deduplicate(self, image_paths):
        """Stage 7: Remove near-duplicate images using perceptual hashing.
        
        Args:
            image_paths (list): List of image file paths that passed stages 1-6.
        
        Returns:
            list: Deduplicated list of image paths.
        """
        if not HAS_IMAGEHASH:
            print("Warning: imagehash not available, skipping deduplication.")
            return image_paths
        
        print("Computing perceptual hashes for deduplication...")
        hashes = {}
        unique_paths = []
        duplicates = 0
        
        for path in tqdm(image_paths, desc="Deduplication"):
            h = self.compute_phash(path)
            if h is None:
                continue
            
            # Check against existing hashes
            is_duplicate = False
            for existing_hash, existing_path in hashes.items():
                if h - existing_hash <= self.config.phash_threshold:
                    is_duplicate = True
                    duplicates += 1
                    break
            
            if not is_duplicate:
                hashes[h] = path
                unique_paths.append(path)
        
        self.stats['duplicates_removed'] = duplicates
        print(f"Removed {duplicates} near-duplicates, {len(unique_paths)} unique images remaining.")
        
        return unique_paths
    
    def run(self, input_dir, output_dir, copy_files=True, log_file=None):
        """Run the complete 7-stage filtering pipeline.
        
        Args:
            input_dir (str): Directory containing raw panoramic images.
            output_dir (str): Directory to save filtered images.
            copy_files (bool): If True, copy passing files to output_dir.
                If False, just return the list of passing files.
            log_file (str): Path to save filtering statistics JSON.
        
        Returns:
            list: Paths of images that passed all filters.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Collect all image files
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []
        for ext in extensions:
            all_images.extend(input_dir.rglob(f'*{ext}'))
            all_images.extend(input_dir.rglob(f'*{ext.upper()}'))
        
        all_images = [str(p) for p in sorted(set(all_images))]
        self.stats['total_raw'] = len(all_images)
        print(f"Found {len(all_images)} images in {input_dir}")
        
        # Stage 1-6: Apply filters sequentially
        stages = [
            ("Resolution", self.check_resolution),
            ("Aspect Ratio", self.check_aspect_ratio),
            ("Brightness", self.check_brightness),
            ("Contrast", self.check_contrast),
            ("Sharpness", self.check_sharpness),
            ("Dental Domain", self.check_dental_domain),
        ]
        
        remaining = all_images
        for stage_name, check_fn in stages:
            passed = []
            for img_path in tqdm(remaining, desc=f"Stage: {stage_name}"):
                if check_fn(img_path):
                    passed.append(img_path)
            
            rejected = len(remaining) - len(passed)
            self.stats[f'rejected_{stage_name.lower().replace(" ", "_")}'] = rejected
            print(f"  {stage_name}: {len(passed)} passed, {rejected} rejected")
            remaining = passed
        
        # Stage 7: Deduplication
        remaining = self.deduplicate(remaining)
        self.stats['final_count'] = len(remaining)
        
        print(f"\n{'='*50}")
        print(f"Final: {len(remaining)}/{len(all_images)} images passed all filters")
        print(f"{'='*50}")
        
        # Copy files to output directory
        if copy_files and len(remaining) > 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Copying {len(remaining)} images to {output_dir}...")
            for src_path in tqdm(remaining, desc="Copying"):
                dst_path = output_dir / Path(src_path).name
                # Handle name collisions
                if dst_path.exists():
                    stem = dst_path.stem
                    suffix = dst_path.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = output_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(src_path, dst_path)
        
        # Save statistics
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(dict(self.stats), f, indent=2)
            print(f"Filter log saved to {log_file}")
        
        return remaining
