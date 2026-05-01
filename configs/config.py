"""
Configuration for Modified DenseCL - Idiopathic Osteosclerosis Detection
=========================================================================
Centralized configuration for all hyperparameters, paths, and ablation flags.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathConfig:
    """Dataset and output paths."""
    # --- Unlabeled data for SSL pretraining ---
    unlabeled_raw_dir: str = "data/unlabeled/raw"          # Raw panoramic images before filtering
    unlabeled_filtered_dir: str = "data/unlabeled/filtered" # After 7-stage filtering
    
    # --- Labeled data for fine-tuning (799 images) ---
    labeled_image_dir: str = "data/labeled/images"
    labeled_annotation_file: str = "data/labeled/annotations.json"  # COCO format
    
    # --- Output ---
    output_dir: str = "output"
    pretrain_checkpoint_dir: str = "output/pretrain/checkpoints"
    finetune_checkpoint_dir: str = "output/finetune/checkpoints"
    tensorboard_dir: str = "output/tensorboard"
    filtered_log_file: str = "output/filter_log.json"


@dataclass
class DataFilterConfig:
    """7-stage data filtering parameters."""
    # Stage 1: Resolution
    min_width: int = 800
    min_height: int = 400
    
    # Stage 2: Aspect ratio (panoramic = wide landscape)
    min_aspect_ratio: float = 1.5   # width/height
    max_aspect_ratio: float = 3.5
    
    # Stage 3: Brightness (mean pixel intensity, 0-255)
    min_brightness: float = 30.0
    max_brightness: float = 220.0
    
    # Stage 4: Contrast (std of pixel intensity)
    min_contrast: float = 20.0
    
    # Stage 5: Sharpness (Laplacian variance)
    min_sharpness: float = 10.0
    
    # Stage 6: Dental domain check
    # Heuristic: dental panoramic X-rays have specific intensity distribution
    dental_intensity_low_ratio: float = 0.05   # max 5% very dark pixels
    dental_intensity_high_ratio: float = 0.05  # max 5% very bright pixels
    
    # Stage 7: Deduplication (perceptual hash)
    phash_threshold: int = 8  # hamming distance threshold


@dataclass
class AugmentationConfig:
    """Augmentation parameters for pretraining."""
    # --- Image size (LANDSCAPE for panoramic) ---
    global_crop_size: Tuple[int, int] = (256, 512)   # (H, W) - landscape
    local_crop_size: Tuple[int, int] = (192, 384)     # Slightly smaller local crop
    
    # --- Standard augmentations ---
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.1   # Low for X-ray (grayscale-like)
    color_jitter_hue: float = 0.02         # Very low for X-ray
    color_jitter_prob: float = 0.8
    
    grayscale_prob: float = 0.2
    gaussian_blur_prob: float = 0.5
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    horizontal_flip_prob: float = 0.5
    
    # --- Asymmetric augmentation (Modification 2) ---
    lesion_crop_bias: float = 0.5         # 50% chance to bias crop to lesion region
    context_padding_factor: float = 2.0   # Context padding = 2x bbox size
    
    # --- Normalization (ImageNet defaults, adjust for dental X-ray if needed) ---
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class PseudoLesionConfig:
    """Parameters for pseudo-lesion detection in unlabeled data."""
    # IO lesions appear as dense (bright) focal regions in bone
    intensity_percentile_threshold: float = 85.0  # Top 15% intensity
    min_area: int = 100        # Minimum lesion area in pixels
    max_area: int = 10000      # Maximum lesion area in pixels
    min_circularity: float = 0.3  # IO lesions tend to be somewhat round
    morphology_kernel_size: int = 5
    adaptive_block_size: int = 51
    adaptive_c: float = -5.0


@dataclass
class DenseCLConfig:
    """DenseCL architecture and training parameters."""
    # --- Backbone ---
    backbone: str = "resnet50"
    backbone_out_channels: int = 2048
    
    # --- Neck ---
    neck_hid_channels: int = 2048
    neck_out_channels: int = 128    # feat_dim
    num_grid: Optional[int] = None  # None = use backbone spatial size
    
    # --- Head ---
    temperature: float = 0.2
    
    # --- Queue ---
    queue_len: int = 65536
    feat_dim: int = 128
    momentum: float = 0.999
    
    # --- Loss ---
    loss_lambda: float = 0.5  # Weight between single and dense loss


@dataclass
class LesionWeightingConfig:
    """Modification 1: Soft Lesion-Aware Weighting."""
    enabled: bool = True
    alpha: float = 2.0                # Weight multiplier for lesion regions
    gaussian_sigma: float = 3.0       # Sigma for Gaussian-smoothed boundary
    background_weight: float = 1.0    # Weight for non-lesion regions


@dataclass
class HardNegativeConfig:
    """Modification 3: Radiologically-Informed Hard Negative Mining."""
    enabled: bool = True
    warmup_epochs: int = 20           # Epochs before activating hard negatives
    top_k: int = 5                    # Number of hard negatives per query
    feature_bank_size: int = 10000    # Max stored features
    intensity_weight: float = 0.3     # Weight for pixel intensity similarity
    feature_weight: float = 0.7       # Weight for cosine similarity in feature space
    update_interval: int = 1          # Update feature bank every N epochs
    hard_neg_loss_weight: float = 0.1 # Additional loss weight for hard negatives


@dataclass
class PretrainConfig:
    """Pretraining hyperparameters."""
    epochs: int = 200
    batch_size: int = 32
    num_workers: int = 4
    
    # --- Optimizer ---
    optimizer: str = "SGD"
    lr: float = 0.03
    weight_decay: float = 1e-4
    sgd_momentum: float = 0.9
    
    # --- Scheduler ---
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 0.0
    
    # --- Checkpointing ---
    checkpoint_interval: int = 20
    log_interval: int = 10      # Log every N iterations
    
    # --- Mixed precision ---
    use_amp: bool = True
    
    # --- Resume ---
    resume_checkpoint: Optional[str] = None


@dataclass
class FinetuneConfig:
    """Faster R-CNN fine-tuning hyperparameters."""
    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    
    # --- Optimizer ---
    lr: float = 0.005
    weight_decay: float = 1e-4
    sgd_momentum: float = 0.9
    
    # --- Scheduler ---
    lr_step_size: int = 15
    lr_gamma: float = 0.1
    
    # --- Model ---
    num_classes: int = 2  # background + IO
    
    # --- Input size ---
    min_size: int = 512
    max_size: int = 1024
    
    # --- Train/Val split ---
    val_ratio: float = 0.2
    
    # --- Pretrained backbone path ---
    pretrained_backbone_path: Optional[str] = None
    
    # --- Checkpointing ---
    checkpoint_interval: int = 5


@dataclass
class AblationConfig:
    """Flags to enable/disable each modification for ablation study."""
    use_lesion_weighting: bool = True       # Modification 1
    use_asymmetric_aug: bool = True         # Modification 2
    use_hard_negative_mining: bool = True   # Modification 3


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    data_filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    pseudo_lesion: PseudoLesionConfig = field(default_factory=PseudoLesionConfig)
    densecl: DenseCLConfig = field(default_factory=DenseCLConfig)
    lesion_weighting: LesionWeightingConfig = field(default_factory=LesionWeightingConfig)
    hard_negative: HardNegativeConfig = field(default_factory=HardNegativeConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    
    # --- Device ---
    device: str = "cuda"
    seed: int = 42
    
    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.paths.output_dir, exist_ok=True)
        os.makedirs(self.paths.pretrain_checkpoint_dir, exist_ok=True)
        os.makedirs(self.paths.finetune_checkpoint_dir, exist_ok=True)
        os.makedirs(self.paths.tensorboard_dir, exist_ok=True)
        os.makedirs(self.paths.unlabeled_filtered_dir, exist_ok=True)


def get_config(**overrides) -> Config:
    """Get config with optional overrides.
    
    Usage:
        cfg = get_config()
        cfg = get_config(pretrain=PretrainConfig(epochs=100))
    """
    cfg = Config(**overrides)
    return cfg


# =========================================================================
# Preset configurations for ablation study
# =========================================================================

def get_baseline_config() -> Config:
    """Vanilla DenseCL (no modifications)."""
    return Config(
        ablation=AblationConfig(
            use_lesion_weighting=False,
            use_asymmetric_aug=False,
            use_hard_negative_mining=False
        ),
        lesion_weighting=LesionWeightingConfig(enabled=False),
        hard_negative=HardNegativeConfig(enabled=False),
    )


def get_mod1_config() -> Config:
    """DenseCL + Soft Lesion-Aware Weighting only."""
    return Config(
        ablation=AblationConfig(
            use_lesion_weighting=True,
            use_asymmetric_aug=False,
            use_hard_negative_mining=False
        ),
        hard_negative=HardNegativeConfig(enabled=False),
    )


def get_mod2_config() -> Config:
    """DenseCL + Asymmetric Augmentation only."""
    return Config(
        ablation=AblationConfig(
            use_lesion_weighting=False,
            use_asymmetric_aug=True,
            use_hard_negative_mining=False
        ),
        lesion_weighting=LesionWeightingConfig(enabled=False),
        hard_negative=HardNegativeConfig(enabled=False),
    )


def get_mod3_config() -> Config:
    """DenseCL + Hard Negative Mining only."""
    return Config(
        ablation=AblationConfig(
            use_lesion_weighting=False,
            use_asymmetric_aug=False,
            use_hard_negative_mining=True
        ),
        lesion_weighting=LesionWeightingConfig(enabled=False),
    )


def get_full_config() -> Config:
    """DenseCL + all 3 modifications."""
    return Config(
        ablation=AblationConfig(
            use_lesion_weighting=True,
            use_asymmetric_aug=True,
            use_hard_negative_mining=True
        ),
    )
