"""
Pretraining Dataset for Modified DenseCL
==========================================
Dataset for self-supervised pretraining on unlabeled dental panoramic images.
Returns dual views (View 1: global, View 2: local/lesion-biased) and 
optional lesion weight maps.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import AsymmetricDualView, PseudoLesionDetector


class DentalPanoramicPretrainDataset(Dataset):
    """Dataset for SSL pretraining on unlabeled dental panoramic images.
    
    Each __getitem__ returns two augmented views of the same image,
    implementing the asymmetric augmentation strategy (Modification 2).
    
    Args:
        image_dir (str): Directory containing filtered panoramic images.
        global_size (tuple): (H, W) for View 1 (global crop).
        local_size (tuple): (H, W) for View 2 (local crop).
        use_asymmetric_aug (bool): Enable asymmetric augmentation.
        use_lesion_weighting (bool): Compute and return lesion weight maps.
        lesion_crop_bias (float): Probability of biasing View 2 to lesion.
        context_padding_factor (float): Context padding for lesion crops.
        augmentation_config (AugmentationConfig): Full augmentation config.
        pseudo_lesion_config (PseudoLesionConfig): Pseudo-lesion detection config.
    """
    
    def __init__(self, 
                 image_dir,
                 global_size=(256, 512),
                 local_size=(192, 384),
                 use_asymmetric_aug=True,
                 use_lesion_weighting=True,
                 lesion_crop_bias=0.5,
                 context_padding_factor=2.0,
                 augmentation_config=None,
                 pseudo_lesion_config=None):
        
        self.image_dir = Path(image_dir)
        
        # Collect image paths
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"[PretrainDataset] Found {len(self.image_paths)} images in {image_dir}")
        
        # Build pseudo-lesion detector
        if pseudo_lesion_config is not None:
            pseudo_detector = PseudoLesionDetector(
                intensity_percentile=pseudo_lesion_config.intensity_percentile_threshold,
                min_area=pseudo_lesion_config.min_area,
                max_area=pseudo_lesion_config.max_area,
                min_circularity=pseudo_lesion_config.min_circularity,
                morphology_kernel=pseudo_lesion_config.morphology_kernel_size,
                adaptive_block_size=pseudo_lesion_config.adaptive_block_size,
                adaptive_c=pseudo_lesion_config.adaptive_c,
            )
        else:
            pseudo_detector = PseudoLesionDetector()
        
        # Build augmentation params
        color_jitter_params = None
        normalize_mean = (0.485, 0.456, 0.406)
        normalize_std = (0.229, 0.224, 0.225)
        grayscale_prob = 0.2
        blur_prob = 0.5
        blur_sigma = (0.1, 2.0)
        flip_prob = 0.5
        
        if augmentation_config is not None:
            color_jitter_params = dict(
                brightness=augmentation_config.color_jitter_brightness,
                contrast=augmentation_config.color_jitter_contrast,
                saturation=augmentation_config.color_jitter_saturation,
                hue=augmentation_config.color_jitter_hue,
            )
            normalize_mean = augmentation_config.normalize_mean
            normalize_std = augmentation_config.normalize_std
            grayscale_prob = augmentation_config.grayscale_prob
            blur_prob = augmentation_config.gaussian_blur_prob
            blur_sigma = augmentation_config.gaussian_blur_sigma
            flip_prob = augmentation_config.horizontal_flip_prob
        
        # Build dual-view transform
        self.transform = AsymmetricDualView(
            global_size=global_size,
            local_size=local_size,
            use_asymmetric=use_asymmetric_aug,
            lesion_crop_bias=lesion_crop_bias,
            context_padding_factor=context_padding_factor,
            pseudo_detector=pseudo_detector,
            color_jitter_params=color_jitter_params,
            grayscale_prob=grayscale_prob,
            blur_prob=blur_prob,
            blur_sigma=blur_sigma,
            flip_prob=flip_prob,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        
        self.use_lesion_weighting = use_lesion_weighting
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get dual-view augmented pair.
        
        Returns:
            dict with keys:
                'view1': Tensor (3, H_global, W_global) — global crop
                'view2': Tensor (3, H_local, W_local) — local/biased crop
                'weight_map': Tensor (H_feat, W_feat) or None — lesion weights
                'idx': int — sample index
        """
        img_path = str(self.image_paths[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Fallback to a random different image
            fallback_idx = (idx + 1) % len(self)
            return self.__getitem__(fallback_idx)
        
        # Apply asymmetric dual-view transform
        view1, view2, weight_map = self.transform(image)
        
        result = {
            'view1': view1,
            'view2': view2,
            'idx': idx,
        }
        
        if self.use_lesion_weighting and weight_map is not None:
            result['weight_map'] = weight_map
        else:
            result['weight_map'] = None
        
        return result


def pretrain_collate_fn(batch):
    """Custom collate function for pretraining dataset.
    
    Handles the case where weight_maps may be None.
    """
    view1s = torch.stack([item['view1'] for item in batch])
    view2s = torch.stack([item['view2'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])
    
    # Handle weight maps (may be None)
    weight_maps = [item['weight_map'] for item in batch]
    if all(w is not None for w in weight_maps):
        weight_maps = torch.stack(weight_maps)
    else:
        weight_maps = None
    
    return {
        'view1': view1s,
        'view2': view2s,
        'weight_map': weight_maps,
        'idx': indices,
    }
