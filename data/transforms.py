"""
Custom Transforms for Modified DenseCL
========================================
Implements:
  - PseudoLesionDetector: Heuristic IO-like region detection for unlabeled data
  - AsymmetricDualView: Asymmetric augmentation (View1=global, View2=local biased)
  - LesionBiasedCrop: Crop biased toward lesion-like dense regions
  - Standard augmentations tuned for dental radiographs
"""

import random
import math
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


class PseudoLesionDetector:
    """Detect pseudo-lesion (IO-like) regions in unlabeled panoramic images.
    
    IO lesions appear as focal dense (bright) sclerotic regions in bone.
    This detector uses intensity-based heuristics to find such regions
    WITHOUT using any ground truth labels (pure SSL).
    
    Args:
        intensity_percentile (float): Percentile threshold for dense regions.
        min_area (int): Minimum region area in pixels.
        max_area (int): Maximum region area in pixels.
        min_circularity (float): Minimum circularity (0-1) of detected region.
        morphology_kernel (int): Kernel size for morphological operations.
        adaptive_block_size (int): Block size for adaptive thresholding.
        adaptive_c (float): Constant subtracted from mean in adaptive thresholding.
    """
    
    def __init__(self, intensity_percentile=85.0, min_area=100, max_area=10000,
                 min_circularity=0.3, morphology_kernel=5, 
                 adaptive_block_size=51, adaptive_c=-5.0):
        self.intensity_percentile = intensity_percentile
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.morphology_kernel = morphology_kernel
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
    
    def detect(self, image):
        """Detect pseudo-lesion regions in an image.
        
        Args:
            image (PIL.Image or np.ndarray): Input dental panoramic image.
        
        Returns:
            list: List of bounding boxes [(x1, y1, x2, y2), ...] for detected regions.
            np.ndarray: Binary mask of detected regions (H, W), values 0 or 255.
        """
        # Convert to grayscale numpy
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        H, W = img_gray.shape
        
        # Step 1: Threshold for dense (bright) regions
        threshold = np.percentile(img_gray, self.intensity_percentile)
        binary = (img_gray > threshold).astype(np.uint8) * 255
        
        # Step 2: Adaptive thresholding to find locally dense regions
        block_size = self.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        if block_size > min(H, W):
            block_size = min(H, W)
            if block_size % 2 == 0:
                block_size -= 1
            block_size = max(block_size, 3)
        
        adaptive = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, self.adaptive_c
        )
        
        # Combine global and adaptive thresholds
        combined = cv2.bitwise_and(binary, adaptive)
        
        # Step 3: Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel, self.morphology_kernel)
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Step 4: Find contours and filter by size/circularity
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bboxes = []
        mask = np.zeros_like(img_gray)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter
            if area < self.min_area or area > self.max_area:
                continue
            
            # Circularity filter (IO lesions tend to be round-ish)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w, y + h))
            
            # Draw on mask
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return bboxes, mask
    
    def create_weight_map(self, image, alpha=2.0, sigma=3.0):
        """Create a lesion-aware weight map for the image.
        
        Args:
            image (PIL.Image or np.ndarray): Input image.
            alpha (float): Weight multiplier for lesion regions.
            sigma (float): Gaussian smoothing sigma for boundary.
        
        Returns:
            np.ndarray: Weight map (H, W), float32. 
                Background = 1.0, Lesion regions = up to alpha.
        """
        bboxes, mask = self.detect(image)
        
        if isinstance(image, Image.Image):
            H, W = image.size[1], image.size[0]
        else:
            H, W = image.shape[:2]
        
        # Start with uniform weight
        weight_map = np.ones((H, W), dtype=np.float32)
        
        if len(bboxes) == 0:
            return weight_map
        
        # Create lesion mask (normalized to 0-1)
        lesion_mask = (mask > 0).astype(np.float32)
        
        # Apply Gaussian smoothing to create soft boundary
        smoothed = gaussian_filter(lesion_mask, sigma=sigma)
        
        # Normalize smoothed mask to [0, 1]
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        
        # Apply alpha weighting: weight = 1.0 + (alpha - 1.0) * smoothed_mask
        weight_map = 1.0 + (alpha - 1.0) * smoothed
        
        return weight_map


class LesionBiasedCrop:
    """Crop biased toward lesion-like regions with context padding.
    
    With probability `bias_prob`, crops around a detected pseudo-lesion region
    with context_padding_factor × bbox size. Otherwise, falls back to 
    standard random crop.
    
    Args:
        output_size (tuple): (H, W) of output crop.
        bias_prob (float): Probability of biasing to lesion region.
        context_padding_factor (float): Context padding multiplier (e.g., 2.0 = 2× bbox).
        pseudo_detector (PseudoLesionDetector): Detector for finding lesion-like regions.
    """
    
    def __init__(self, output_size=(192, 384), bias_prob=0.5,
                 context_padding_factor=2.0, pseudo_detector=None):
        self.output_size = output_size  # (H, W)
        self.bias_prob = bias_prob
        self.context_padding_factor = context_padding_factor
        self.pseudo_detector = pseudo_detector or PseudoLesionDetector()
    
    def __call__(self, image):
        """Apply lesion-biased cropping.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Cropped image.
        """
        W_img, H_img = image.size
        H_out, W_out = self.output_size
        
        if random.random() < self.bias_prob:
            # Try to find pseudo-lesion regions
            bboxes, _ = self.pseudo_detector.detect(image)
            
            if len(bboxes) > 0:
                # Randomly select one lesion region
                bbox = random.choice(bboxes)
                x1, y1, x2, y2 = bbox
                
                # Compute center and size
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                
                # Apply context padding
                crop_w = max(int(bw * self.context_padding_factor), W_out)
                crop_h = max(int(bh * self.context_padding_factor), H_out)
                
                # Center crop around lesion with some jitter
                jitter_x = random.randint(-int(bw * 0.3), int(bw * 0.3))
                jitter_y = random.randint(-int(bh * 0.3), int(bh * 0.3))
                
                crop_x1 = max(0, int(cx - crop_w / 2 + jitter_x))
                crop_y1 = max(0, int(cy - crop_h / 2 + jitter_y))
                crop_x2 = min(W_img, crop_x1 + crop_w)
                crop_y2 = min(H_img, crop_y1 + crop_h)
                
                # Adjust if crop goes out of bounds
                if crop_x2 - crop_x1 < W_out:
                    crop_x1 = max(0, crop_x2 - W_out)
                if crop_y2 - crop_y1 < H_out:
                    crop_y1 = max(0, crop_y2 - H_out)
                
                # Crop and resize
                cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                cropped = cropped.resize((W_out, H_out), Image.BILINEAR)
                return cropped
        
        # Fallback: standard random crop
        if W_img < W_out or H_img < H_out:
            image = image.resize((max(W_img, W_out), max(H_img, H_out)), Image.BILINEAR)
            W_img, H_img = image.size
        
        x = random.randint(0, W_img - W_out)
        y = random.randint(0, H_img - H_out)
        cropped = image.crop((x, y, x + W_out, y + H_out))
        
        return cropped


class GaussianBlur:
    """Gaussian blur augmentation (as in MoCo/DenseCL)."""
    
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class AsymmetricDualView:
    """Generate asymmetric dual views for Modified DenseCL.
    
    View 1 (Global): Full panoramic crop with standard augmentations.
    View 2 (Local):  Lesion-biased local crop (50% of the time) with 
                     context padding 2× bounding box size.
    
    This implements Modification 2: Asymmetric Augmentation.
    
    Args:
        global_size (tuple): (H, W) for global view.
        local_size (tuple): (H, W) for local view.
        use_asymmetric (bool): If False, both views use standard global cropping.
        lesion_crop_bias (float): Probability of biasing local crop to lesion.
        context_padding_factor (float): Context padding for lesion crops.
        pseudo_detector (PseudoLesionDetector): Pseudo-lesion detector.
        color_jitter_params (dict): ColorJitter parameters.
        grayscale_prob (float): Probability of converting to grayscale.
        blur_prob (float): Probability of Gaussian blur.
        flip_prob (float): Probability of horizontal flip.
        normalize_mean (tuple): Normalization mean.
        normalize_std (tuple): Normalization std.
    """
    
    def __init__(self, 
                 global_size=(256, 512),
                 local_size=(192, 384),
                 use_asymmetric=True,
                 lesion_crop_bias=0.5,
                 context_padding_factor=2.0,
                 pseudo_detector=None,
                 color_jitter_params=None,
                 grayscale_prob=0.2,
                 blur_prob=0.5,
                 blur_sigma=(0.1, 2.0),
                 flip_prob=0.5,
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225)):
        
        self.global_size = global_size  # (H, W)
        self.local_size = local_size
        self.use_asymmetric = use_asymmetric
        
        # Pseudo-lesion detector
        self.pseudo_detector = pseudo_detector or PseudoLesionDetector()
        
        # Lesion-biased cropper for View 2
        self.lesion_cropper = LesionBiasedCrop(
            output_size=local_size,
            bias_prob=lesion_crop_bias,
            context_padding_factor=context_padding_factor,
            pseudo_detector=self.pseudo_detector
        )
        
        # Color augmentations (tuned for dental X-rays)
        if color_jitter_params is None:
            color_jitter_params = dict(
                brightness=0.3, contrast=0.3, 
                saturation=0.1, hue=0.02  # Low sat/hue for X-ray
            )
        
        self.color_jitter = T.ColorJitter(**color_jitter_params)
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur = GaussianBlur(*blur_sigma)
        self.flip_prob = flip_prob
        
        self.normalize = T.Normalize(mean=normalize_mean, std=normalize_std)
        self.to_tensor = T.ToTensor()
    
    def _apply_augmentations(self, img):
        """Apply standard augmentations to a single view."""
        # Color jitter
        if random.random() < 0.8:
            img = self.color_jitter(img)
        
        # Random grayscale
        if random.random() < self.grayscale_prob:
            img = TF.to_grayscale(img, num_output_channels=3)
        
        # Gaussian blur
        if random.random() < self.blur_prob:
            img = self.blur(img)
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            img = TF.hflip(img)
        
        return img
    
    def __call__(self, image):
        """Generate two views from the same image.
        
        Args:
            image (PIL.Image): Input panoramic image.
        
        Returns:
            tuple: (view1_tensor, view2_tensor, weight_map_tensor)
                - view1: (3, H_global, W_global)
                - view2: (3, H_local, W_local) 
                - weight_map: (H_global, W_global) or None
        """
        H_g, W_g = self.global_size
        
        # ===== View 1: Global crop =====
        # Resize maintaining aspect then crop, or just resize
        view1 = image.resize((W_g, H_g), Image.BILINEAR)
        
        # Compute weight map BEFORE augmentation (on original view)
        weight_map = None
        if self.use_asymmetric:
            weight_map = self.pseudo_detector.create_weight_map(
                view1, alpha=2.0, sigma=3.0
            )
            weight_map = torch.from_numpy(weight_map).float()
        
        view1 = self._apply_augmentations(view1)
        view1 = self.to_tensor(view1)
        view1 = self.normalize(view1)
        
        # ===== View 2: Local or global crop =====
        if self.use_asymmetric:
            # Lesion-biased local crop
            view2 = self.lesion_cropper(image)
        else:
            # Standard global crop (same as View 1)
            H_l, W_l = self.local_size
            view2 = image.resize((W_l, H_l), Image.BILINEAR)
        
        view2 = self._apply_augmentations(view2)
        view2 = self.to_tensor(view2)
        view2 = self.normalize(view2)
        
        return view1, view2, weight_map
