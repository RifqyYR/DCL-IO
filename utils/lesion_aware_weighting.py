"""
Soft Lesion-Aware Weighting (Modification 1)
==============================================
Dense contrastive loss is weighted higher (alpha=2.0) on spatial positions
corresponding to lesion regions, with Gaussian-smoothed boundaries.

This encourages the model to learn better representations for IO lesion regions
while maintaining global context (background weight = 1.0, not zeroed out).

Key design: Gaussian smoothing ensures gradual weight transition at lesion
boundaries, avoiding sharp discontinuities that could cause training instability.
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


class LesionAwareWeightComputer:
    """Compute lesion-aware weight maps for dense contrastive loss.
    
    The weight map has value 1.0 for background (normal bone, teeth, etc.)
    and up to `alpha` for lesion regions. Gaussian smoothing creates a soft
    boundary transition.
    
    Args:
        alpha (float): Maximum weight for lesion center. Default 2.0.
        sigma (float): Gaussian smoothing sigma. Default 3.0.
        background_weight (float): Weight for non-lesion regions. Default 1.0.
    """
    
    def __init__(self, alpha=2.0, sigma=3.0, background_weight=1.0):
        self.alpha = alpha
        self.sigma = sigma
        self.background_weight = background_weight
    
    def compute_from_mask(self, lesion_mask):
        """Compute weight map from a binary lesion mask.
        
        Args:
            lesion_mask (np.ndarray): Binary mask (H, W), values 0 or 1.
        
        Returns:
            np.ndarray: Weight map (H, W), float32.
                Background = background_weight, Lesion center = alpha.
        """
        # Start with binary mask
        mask_float = lesion_mask.astype(np.float32)
        
        # Apply Gaussian smoothing for soft boundary
        smoothed = gaussian_filter(mask_float, sigma=self.sigma)
        
        # Normalize to [0, 1]
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        
        # Map to weight range: background_weight + (alpha - background_weight) * smoothed
        weight_map = self.background_weight + (self.alpha - self.background_weight) * smoothed
        
        return weight_map
    
    def compute_from_bboxes(self, bboxes, image_size):
        """Compute weight map from bounding boxes.
        
        Creates elliptical lesion masks centered on each bbox,
        then applies Gaussian smoothing.
        
        Args:
            bboxes (list): List of (x1, y1, x2, y2) bounding boxes.
            image_size (tuple): (H, W) of the image.
        
        Returns:
            np.ndarray: Weight map (H, W), float32.
        """
        H, W = image_size
        mask = np.zeros((H, W), dtype=np.float32)
        
        for x1, y1, x2, y2 in bboxes:
            # Create elliptical mask centered on bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            rx = (x2 - x1) / 2
            ry = (y2 - y1) / 2
            
            if rx <= 0 or ry <= 0:
                continue
            
            # Generate coordinate grid
            y_coords, x_coords = np.ogrid[0:H, 0:W]
            
            # Elliptical distance
            ellipse = ((x_coords - cx) / max(rx, 1e-6))**2 + \
                      ((y_coords - cy) / max(ry, 1e-6))**2
            
            # Inside ellipse = 1.0
            mask = np.maximum(mask, (ellipse <= 1.0).astype(np.float32))
        
        return self.compute_from_mask(mask)
    
    def resize_weight_map(self, weight_map, target_size):
        """Resize weight map to match feature map spatial dimensions.
        
        Args:
            weight_map (torch.Tensor or np.ndarray): Weight map (H, W).
            target_size (tuple): Target (H_feat, W_feat).
        
        Returns:
            torch.Tensor: Resized weight map (H_feat, W_feat).
        """
        if isinstance(weight_map, np.ndarray):
            weight_map = torch.from_numpy(weight_map).float()
        
        # Add batch and channel dims for interpolation
        wm = weight_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        resized = F.interpolate(wm, size=target_size, mode='bilinear', 
                               align_corners=False)
        return resized.squeeze(0).squeeze(0)  # (H_feat, W_feat)
    
    def prepare_dense_weights(self, weight_maps, spatial_h, spatial_w):
        """Prepare weight vector for dense contrastive loss.
        
        Resizes weight maps to match feature spatial dims and flattens 
        for element-wise multiplication with dense loss.
        
        Args:
            weight_maps (torch.Tensor): Batch of weight maps (B, H, W).
            spatial_h (int): Height of feature map.
            spatial_w (int): Width of feature map.
        
        Returns:
            torch.Tensor: Flattened weights (B * spatial_h * spatial_w,).
        """
        B = weight_maps.shape[0]
        
        # Resize each weight map to feature spatial dims
        wm = weight_maps.unsqueeze(1)  # (B, 1, H, W)
        resized = F.interpolate(wm, size=(spatial_h, spatial_w), 
                               mode='bilinear', align_corners=False)
        resized = resized.squeeze(1)  # (B, H_feat, W_feat)
        
        # Flatten: (B * H_feat * W_feat,)
        return resized.reshape(-1)
