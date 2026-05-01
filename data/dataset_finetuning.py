"""
Fine-tuning Dataset for Faster R-CNN
======================================
Loads labeled dental panoramic images with COCO-format annotations
for Faster R-CNN training and evaluation.
"""

import os
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class DentalPanoramicDetectionDataset(Dataset):
    """Dataset for object detection fine-tuning on labeled dental panoramic images.
    
    Expects COCO-format annotations:
    {
        "images": [{"id": 1, "file_name": "img001.jpg", "width": 2000, "height": 1000}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, 
                         "bbox": [x, y, w, h], "area": ..., "iscrowd": 0}, ...],
        "categories": [{"id": 1, "name": "IO"}, ...]
    }
    
    Args:
        image_dir (str): Directory containing labeled images.
        annotation_file (str): Path to COCO-format JSON annotation file.
        transforms (callable, optional): Augmentation transforms.
        is_train (bool): Whether this is training data (enables augmentation).
    """
    
    def __init__(self, image_dir, annotation_file, transforms=None, is_train=True):
        self.image_dir = Path(image_dir)
        self.is_train = is_train
        self.transforms = transforms
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image info mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Build annotation mapping (image_id -> list of annotations)
        self.annotations = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Category mapping
        self.categories = {cat['id']: cat['name'] 
                          for cat in coco_data.get('categories', [])}
        
        # Image IDs with at least one annotation (for training)
        # For evaluation, keep all images
        self.image_ids = list(self.images.keys())
        
        print(f"[DetectionDataset] Loaded {len(self.image_ids)} images, "
              f"{sum(len(v) for v in self.annotations.values())} annotations, "
              f"{len(self.categories)} categories")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get image with detection targets.
        
        Returns:
            tuple: (image_tensor, target_dict)
                image_tensor: (3, H, W)
                target_dict: {
                    'boxes': (N, 4) in [x1, y1, x2, y2] format,
                    'labels': (N,) class labels (1-indexed, 0 = background),
                    'image_id': int,
                    'area': (N,),
                    'iscrowd': (N,),
                }
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(str(img_path)).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # No annotations — empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
        }
        
        # Apply training augmentations
        if self.is_train and self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # Convert to tensor if not already
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        
        return image, target


class DetectionTransforms:
    """Comprehensive augmentation pipeline for detection training.
    
    Combats overfitting on small datasets (~640 images) through:
    1. Multi-scale random resize (scale jitter)
    2. Random crop with IoU-aware box filtering
    3. Horizontal flip
    4. Photometric distortion (brightness, contrast, saturation)
    5. Gaussian blur / noise
    6. Random erasing (cutout-style occlusion)
    
    All transforms properly adjust bounding boxes.
    
    Args:
        flip_prob (float): Horizontal flip probability.
        multi_scale (list): Scale factors for multi-scale training.
        crop_prob (float): Random crop probability.
        crop_scale (tuple): (min, max) crop scale relative to image.
        color_jitter_prob (float): Probability of color distortion.
        blur_prob (float): Probability of Gaussian blur.
        erase_prob (float): Probability of random erasing.
        min_box_visibility (float): Minimum fraction of box area that must
            remain visible after crop to keep the annotation.
    """
    
    def __init__(self, flip_prob=0.5, 
                 multi_scale=None,
                 crop_prob=0.3,
                 crop_scale=(0.8, 1.0),
                 color_jitter_prob=0.5,
                 blur_prob=0.2,
                 erase_prob=0.15,
                 min_box_visibility=0.5):
        self.flip_prob = flip_prob
        self.multi_scale = multi_scale or [0.9, 0.95, 1.0, 1.05, 1.1]
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.color_jitter_prob = color_jitter_prob
        self.blur_prob = blur_prob
        self.erase_prob = erase_prob
        self.min_box_visibility = min_box_visibility
    
    def _clip_boxes(self, boxes, w, h):
        """Clip boxes to image boundaries."""
        boxes[:, 0] = boxes[:, 0].clamp(0, w)
        boxes[:, 1] = boxes[:, 1].clamp(0, h)
        boxes[:, 2] = boxes[:, 2].clamp(0, w)
        boxes[:, 3] = boxes[:, 3].clamp(0, h)
        return boxes
    
    def _filter_boxes(self, target, min_area=4.0):
        """Remove degenerate boxes (too small after crop/clip)."""
        boxes = target['boxes']
        if len(boxes) == 0:
            return target
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > 1) & (heights > 1) & (widths * heights > min_area)
        
        target['boxes'] = boxes[keep]
        target['labels'] = target['labels'][keep]
        target['area'] = target['area'][keep]
        target['iscrowd'] = target['iscrowd'][keep]
        return target
    
    def _multi_scale_resize(self, image, target):
        """Randomly resize the image by a scale factor."""
        import random
        scale = random.choice(self.multi_scale)
        
        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        if len(target['boxes']) > 0:
            target['boxes'] = target['boxes'] * scale
            target['area'] = target['area'] * (scale ** 2)
        
        return image, target
    
    def _random_crop(self, image, target):
        """Random crop with IoU-aware box filtering.
        
        Ensures at least some annotated region remains visible.
        """
        import random
        
        w, h = image.size
        boxes = target['boxes']
        
        if len(boxes) == 0 or random.random() > self.crop_prob:
            return image, target
        
        # Determine crop size
        min_scale, max_scale = self.crop_scale
        crop_scale = random.uniform(min_scale, max_scale)
        crop_w = int(w * crop_scale)
        crop_h = int(h * crop_scale)
        
        # Try up to 10 random crops, pick one that keeps at least 1 box
        best_crop = None
        best_kept = 0
        
        for _ in range(10):
            x1 = random.randint(0, max(0, w - crop_w))
            y1 = random.randint(0, max(0, h - crop_h))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            # Compute box visibility after crop
            if len(boxes) > 0:
                clipped = boxes.clone()
                clipped[:, 0] = clipped[:, 0].clamp(x1, x2)
                clipped[:, 1] = clipped[:, 1].clamp(y1, y2)
                clipped[:, 2] = clipped[:, 2].clamp(x1, x2)
                clipped[:, 3] = clipped[:, 3].clamp(y1, y2)
                
                clipped_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
                orig_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                visibility = clipped_area / orig_area.clamp(min=1e-6)
                
                kept = (visibility >= self.min_box_visibility).sum().item()
            else:
                kept = 0
            
            if kept > best_kept:
                best_kept = kept
                best_crop = (x1, y1, x2, y2)
            
            if kept >= len(boxes):
                break  # All boxes are kept
        
        if best_crop is None or best_kept == 0:
            return image, target
        
        x1, y1, x2, y2 = best_crop
        
        # Crop image
        image = image.crop((x1, y1, x2, y2))
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, 0] -= x1
            boxes[:, 1] -= y1
            boxes[:, 2] -= x1
            boxes[:, 3] -= y1
            
            # Clip to crop boundary
            boxes = self._clip_boxes(boxes, crop_w, crop_h)
            target['boxes'] = boxes
            
            # Recalculate area
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # Filter degenerate boxes
            orig_area = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                        (target['boxes'][:, 3] - target['boxes'][:, 1])
            keep = orig_area > 4.0
            target['boxes'] = target['boxes'][keep]
            target['labels'] = target['labels'][keep]
            target['area'] = target['area'][keep]
            target['iscrowd'] = target['iscrowd'][keep]
        
        return image, target
    
    def _color_jitter(self, image):
        """Photometric distortion tuned for dental X-rays.
        
        X-rays are mostly grayscale, so we focus on brightness and contrast.
        """
        import random
        from PIL import ImageEnhance, ImageFilter
        
        # Brightness (±30%)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Contrast (±40%) — important for IO lesion visibility
        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        # Sharpness (±30%)
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Sharpness(image).enhance(factor)
        
        # Gamma correction
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            img_array = np.array(image).astype(np.float32) / 255.0
            img_array = np.power(img_array, gamma)
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return image
    
    def _gaussian_blur(self, image):
        """Apply random Gaussian blur."""
        import random
        from PIL import ImageFilter
        
        sigma = random.uniform(0.5, 1.5)
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image
    
    def _random_erase(self, image, target):
        """Random erasing (cutout) — occludes random rectangular patches.
        
        This regularization technique forces the model to rely on context
        rather than single discriminative features.
        """
        import random
        
        w, h = image.size
        img_array = np.array(image)
        
        # Erase 1-3 patches
        num_patches = random.randint(1, 3)
        for _ in range(num_patches):
            # Patch size: 5-15% of image dimension
            pw = random.randint(int(w * 0.05), int(w * 0.15))
            ph = random.randint(int(h * 0.05), int(h * 0.15))
            px = random.randint(0, max(0, w - pw))
            py = random.randint(0, max(0, h - ph))
            
            # Fill with mean pixel value (neutral for X-rays)
            mean_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
            img_array[py:py+ph, px:px+pw] = mean_val
        
        image = Image.fromarray(img_array)
        return image, target
    
    def __call__(self, image, target):
        """Apply the full augmentation pipeline.
        
        Args:
            image (PIL.Image): Input image.
            target (dict): Detection targets with 'boxes' key.
        
        Returns:
            tuple: (transformed_image, transformed_target)
        """
        import random
        
        # 1. Multi-scale resize
        image, target = self._multi_scale_resize(image, target)
        
        # 2. Random crop (IoU-aware)
        image, target = self._random_crop(image, target)
        
        # 3. Horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            if len(target['boxes']) > 0:
                w = image.size[0]
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        # 4. Color jitter
        if random.random() < self.color_jitter_prob:
            image = self._color_jitter(image)
        
        # 5. Gaussian blur
        if random.random() < self.blur_prob:
            image = self._gaussian_blur(image)
        
        # 6. Random erasing
        if random.random() < self.erase_prob:
            image, target = self._random_erase(image, target)
        
        # Final filter to remove any degenerate boxes
        target = self._filter_boxes(target)
        
        return image, target


def build_detection_datasets(image_dir, annotation_file, val_ratio=0.2, seed=42):
    """Build train and val detection datasets with train/val split.
    
    Args:
        image_dir (str): Directory containing labeled images.
        annotation_file (str): Path to COCO-format annotation file.
        val_ratio (float): Fraction of data for validation.
        seed (int): Random seed for reproducible split.
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load full annotation
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Split image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    np.random.seed(seed)
    np.random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * (1 - val_ratio))
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    # Create separate annotation files
    train_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in train_ids],
        'annotations': [ann for ann in coco_data.get('annotations', []) 
                       if ann['image_id'] in train_ids],
        'categories': coco_data.get('categories', [])
    }
    
    val_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in val_ids],
        'annotations': [ann for ann in coco_data.get('annotations', []) 
                       if ann['image_id'] in val_ids],
        'categories': coco_data.get('categories', [])
    }
    
    # Save split annotations to temp files
    import tempfile
    train_ann_path = os.path.join(os.path.dirname(annotation_file), 
                                   'annotations_train_split.json')
    val_ann_path = os.path.join(os.path.dirname(annotation_file), 
                                 'annotations_val_split.json')
    
    with open(train_ann_path, 'w') as f:
        json.dump(train_coco, f)
    with open(val_ann_path, 'w') as f:
        json.dump(val_coco, f)
    
    # Build datasets
    train_transforms = DetectionTransforms(flip_prob=0.5)
    
    train_dataset = DentalPanoramicDetectionDataset(
        image_dir=image_dir,
        annotation_file=train_ann_path,
        transforms=train_transforms,
        is_train=True
    )
    
    val_dataset = DentalPanoramicDetectionDataset(
        image_dir=image_dir,
        annotation_file=val_ann_path,
        transforms=None,
        is_train=False
    )
    
    print(f"[Split] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset
