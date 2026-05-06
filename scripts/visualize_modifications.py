"""
Visualize DenseCL Modifications
==================================
Generates visual explanations of the 3 modifications used in
Modified DenseCL pretraining:

  1. Soft Lesion-Aware Weighting — weight map overlay
  2. Asymmetric Augmentation — global vs local crop comparison
  3. Hard Negative Mining — IO-similar normal regions

Usage:
    python scripts/visualize_modifications.py \
        --image_dir data/unlabeled/filtered \
        --output_dir output/visualizations \
        --num_images 5
"""

import argparse
import os
import sys
import random

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import PseudoLesionDetector, LesionBiasedCrop, AsymmetricDualView

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def load_image(path):
    """Load image as PIL RGB."""
    return Image.open(path).convert('RGB')


def get_font(size=14):
    """Get a font, with fallback."""
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except (IOError, OSError):
        return ImageFont.load_default()


# =========================================================================
# Mod 1: Soft Lesion-Aware Weighting
# =========================================================================

def visualize_lesion_weighting(image_path, output_path):
    """Visualize how Soft Lesion-Aware Weighting works.
    
    Creates a figure with:
      (a) Original image
      (b) Detected pseudo-lesion regions (bboxes + mask)
      (c) Raw binary lesion mask
      (d) Gaussian-smoothed weight map
      (e) Weight map overlaid on image
    """
    image = load_image(image_path)
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    detector = PseudoLesionDetector()
    bboxes, mask = detector.detect(image)
    weight_map = detector.create_weight_map(image, alpha=2.0, sigma=3.0)

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle('Modification 1: Soft Lesion-Aware Weighting',
                 fontsize=16, fontweight='bold', y=1.02)

    # (a) Original
    axes[0].imshow(img_np)
    axes[0].set_title('(a) Original Image')
    axes[0].axis('off')

    # (b) Detected pseudo-lesion regions
    img_bbox = img_np.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
    axes[1].imshow(img_bbox)
    axes[1].set_title(f'(b) Pseudo-Lesion Regions ({len(bboxes)} found)')
    axes[1].axis('off')

    # (c) Binary mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('(c) Binary Lesion Mask')
    axes[2].axis('off')

    # (d) Weight map (heatmap)
    cmap = LinearSegmentedColormap.from_list('weight',
        [(0, '#1a237e'), (0.5, '#4fc3f7'), (1, '#ff1744')])
    im = axes[3].imshow(weight_map, cmap=cmap, vmin=1.0, vmax=2.0)
    axes[3].set_title('(d) Gaussian-Smoothed\nWeight Map (σ=3.0)')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04,
                 label='Weight (1.0=bg, 2.0=lesion)')

    # (e) Overlay
    img_float = img_gray.astype(np.float32) / 255.0
    overlay = np.stack([img_float]*3, axis=-1)
    # Tint lesion regions red based on weight
    lesion_intensity = (weight_map - 1.0)  # 0 to 1
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + lesion_intensity * 0.4, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] - lesion_intensity * 0.15, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] - lesion_intensity * 0.15, 0, 1)
    axes[4].imshow(overlay)
    axes[4].set_title('(e) Weight Map Overlay\n(red = higher weight)')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =========================================================================
# Mod 2: Asymmetric Augmentation
# =========================================================================

def visualize_asymmetric_augmentation(image_path, output_path):
    """Visualize how Asymmetric Augmentation works.
    
    Creates a figure with:
      (a) Original image with pseudo-lesion boxes
      (b) View 1: Global crop (full panoramic, 256×512)
      (c-f) View 2: Multiple local crops biased toward lesion regions
    """
    image = load_image(image_path)
    img_np = np.array(image)

    detector = PseudoLesionDetector()
    bboxes, _ = detector.detect(image)

    global_size = (256, 512)
    local_size = (192, 384)

    # Generate multiple local crops to show variety
    cropper = LesionBiasedCrop(
        output_size=local_size, bias_prob=0.7,
        context_padding_factor=2.0, pseudo_detector=detector
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Modification 2: Asymmetric Augmentation',
                 fontsize=16, fontweight='bold', y=1.02)

    # (a) Original with detected regions
    img_bbox = img_np.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
    axes[0, 0].imshow(img_bbox)
    axes[0, 0].set_title(f'(a) Original + Pseudo-Lesions ({len(bboxes)})',
                         fontsize=12)
    axes[0, 0].axis('off')

    # (b) View 1: Global crop
    view1 = image.resize((global_size[1], global_size[0]), Image.BILINEAR)
    axes[0, 1].imshow(np.array(view1))
    axes[0, 1].set_title(f'(b) View 1: Global Crop\n{global_size[0]}×{global_size[1]}',
                         fontsize=12)
    axes[0, 1].set_xlabel('Full panoramic context preserved')
    axes[0, 1].axis('off')

    # Info panel
    axes[0, 2].axis('off')
    info_text = (
        "Asymmetric Dual-View Strategy\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "View 1 (Global):\n"
        f"  • Size: {global_size[0]}×{global_size[1]}\n"
        "  • Full panoramic resize\n"
        "  • Preserves global anatomy\n\n"
        "View 2 (Local):\n"
        f"  • Size: {local_size[0]}×{local_size[1]}\n"
        "  • 50% biased to lesion regions\n"
        "  • Context padding: 2× bbox\n\n"
        "Purpose:\n"
        "  Forces model to match local\n"
        "  lesion features with global\n"
        "  anatomical context"
    )
    axes[0, 2].text(0.1, 0.95, info_text, transform=axes[0, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (c-e) View 2: Local crops (3 samples)
    for i in range(3):
        random.seed(i * 42 + 7)
        local_crop = cropper(image)
        axes[1, i].imshow(np.array(local_crop))
        axes[1, i].set_title(f'({"cde"[i]}) View 2: Local Crop #{i+1}\n'
                             f'{local_size[0]}×{local_size[1]}',
                             fontsize=12)
        axes[1, i].set_xlabel('Lesion-biased crop with context')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =========================================================================
# Mod 3: Hard Negative Mining
# =========================================================================

def visualize_hard_negatives(image_path, output_path):
    """Visualize how Hard Negative Mining identifies confusing regions.
    
    Creates a figure showing:
      (a) Original image
      (b) Pseudo-lesion regions (what the model should learn to detect)
      (c) Normal bone regions with high intensity (potential hard negatives)
      (d) Side-by-side: lesion-like vs actual lesion patches
      (e) Conceptual diagram of hard negative injection
    """
    image = load_image(image_path)
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    H, W = img_gray.shape

    detector = PseudoLesionDetector()
    bboxes, lesion_mask = detector.detect(image)

    # Find hard negative candidates: bright normal bone (NOT in lesion mask)
    threshold_high = np.percentile(img_gray, 80)
    bright_mask = (img_gray > threshold_high).astype(np.uint8) * 255

    # Remove lesion regions from bright mask → remaining = hard negatives
    non_lesion = cv2.bitwise_not(lesion_mask)
    hard_neg_mask = cv2.bitwise_and(bright_mask, non_lesion)

    # Find contours of hard negative regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hard_neg_mask = cv2.morphologyEx(hard_neg_mask, cv2.MORPH_CLOSE, kernel)
    hard_neg_mask = cv2.morphologyEx(hard_neg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(hard_neg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size (similar to IO lesions)
    hn_bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if 200 < area < 15000:
            x, y, w, h = cv2.boundingRect(c)
            hn_bboxes.append((x, y, x+w, y+h))

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Modification 3: Radiologically-Informed Hard Negative Mining',
                 fontsize=16, fontweight='bold', y=1.02)

    # (a) Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title('(a) Original Image')
    ax.axis('off')

    # (b) Lesion regions (green)
    img_lesion = img_np.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_lesion, (x1, y1), (x2, y2), (0, 255, 0), 3)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_lesion)
    ax.set_title(f'(b) IO-Like Regions (green)\n{len(bboxes)} detected')
    ax.axis('off')

    # (c) Hard negative regions (red) — look similar but aren't lesions
    img_hn = img_np.copy()
    for (x1, y1, x2, y2) in hn_bboxes[:20]:
        cv2.rectangle(img_hn, (x1, y1), (x2, y2), (255, 50, 50), 2)
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img_hn)
    ax.set_title(f'(c) Hard Negatives (red)\n{len(hn_bboxes)} candidates')
    ax.axis('off')

    # (d) Combined view
    img_combined = img_np.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for (x1, y1, x2, y2) in hn_bboxes[:15]:
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), (255, 50, 50), 2)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(img_combined)
    ax.set_title('(d) Combined: Lesions vs\nHard Negatives')
    ax.axis('off')

    # (e-f) Patch comparisons: lesion patches vs hard negative patches
    lesion_patches = []
    for (x1, y1, x2, y2) in bboxes[:6]:
        pad = 10
        px1, py1 = max(0, x1-pad), max(0, y1-pad)
        px2, py2 = min(W, x2+pad), min(H, y2+pad)
        patch = img_np[py1:py2, px1:px2]
        if patch.size > 0:
            lesion_patches.append(patch)

    hn_patches = []
    for (x1, y1, x2, y2) in hn_bboxes[:6]:
        pad = 10
        px1, py1 = max(0, x1-pad), max(0, y1-pad)
        px2, py2 = min(W, x2+pad), min(H, y2+pad)
        patch = img_np[py1:py2, px1:px2]
        if patch.size > 0:
            hn_patches.append(patch)

    n_show = min(4, len(lesion_patches), len(hn_patches))
    for i in range(n_show):
        # Lesion patch (top sub-row)
        ax = fig.add_subplot(gs[1, i])
        if i < len(lesion_patches) and i < len(hn_patches):
            # Stack vertically: lesion on top, hard neg on bottom
            lp = cv2.resize(lesion_patches[i], (120, 120))
            hp = cv2.resize(hn_patches[i], (120, 120))
            # Add border
            lp_bordered = cv2.copyMakeBorder(lp, 3, 3, 3, 3,
                                              cv2.BORDER_CONSTANT, value=(0, 200, 0))
            hp_bordered = cv2.copyMakeBorder(hp, 3, 3, 3, 3,
                                              cv2.BORDER_CONSTANT, value=(200, 50, 50))
            # Create separator
            sep = np.ones((10, lp_bordered.shape[1], 3), dtype=np.uint8) * 255
            combined = np.vstack([lp_bordered, sep, hp_bordered])
            ax.imshow(combined)
            ax.set_title(f'Pair #{i+1}', fontsize=11)
            ax.set_ylabel('Top=IO  Bottom=Normal' if i == 0 else '', fontsize=9)
        ax.axis('off')

    if n_show == 0:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, 'No patches available for comparison\n'
                '(no pseudo-lesion or hard negatives found in this image)',
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax.transAxes)
        ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =========================================================================
# Combined overview
# =========================================================================

def visualize_pipeline_overview(image_path, output_path):
    """Create a single overview figure showing all 3 modifications."""
    image = load_image(image_path)
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    detector = PseudoLesionDetector()
    bboxes, mask = detector.detect(image)
    weight_map = detector.create_weight_map(image, alpha=2.0, sigma=3.0)

    global_size = (256, 512)
    local_size = (192, 384)
    cropper = LesionBiasedCrop(output_size=local_size, bias_prob=0.7,
                                pseudo_detector=detector)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Modified DenseCL — Three Modifications Overview',
                 fontsize=18, fontweight='bold', y=1.02)

    # Row 0: Original + 3 mods
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    # Mod 1: Weight map
    cmap = LinearSegmentedColormap.from_list('w', ['#1a237e', '#4fc3f7', '#ff1744'])
    axes[0, 1].imshow(weight_map, cmap=cmap, vmin=1.0, vmax=2.0)
    axes[0, 1].set_title('Mod 1: Lesion-Aware\nWeight Map', fontsize=12,
                         color='#1565c0')
    axes[0, 1].axis('off')

    # Mod 2: Global vs Local
    view1 = image.resize((global_size[1], global_size[0]))
    axes[0, 2].imshow(np.array(view1))
    axes[0, 2].set_title('Mod 2: View 1 (Global)\n256×512', fontsize=12,
                         color='#2e7d32')
    axes[0, 2].axis('off')

    random.seed(42)
    view2 = cropper(image)
    axes[0, 3].imshow(np.array(view2))
    axes[0, 3].set_title('Mod 2: View 2 (Local)\n192×384 (lesion-biased)',
                         fontsize=12, color='#2e7d32')
    axes[0, 3].axis('off')

    # Row 1: Details
    # Mod 1 detail: overlay
    img_float = img_gray.astype(np.float32) / 255.0
    overlay = np.stack([img_float]*3, axis=-1)
    li = (weight_map - 1.0)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + li * 0.5, 0, 1)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Mod 1: Weight Overlay\n(red = α=2.0)', fontsize=11)
    axes[1, 0].axis('off')

    # Mod 1 detail: detected regions
    img_bbox = img_np.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
    axes[1, 1].imshow(img_bbox)
    axes[1, 1].set_title(f'Detected Pseudo-Lesions\n({len(bboxes)} regions)',
                         fontsize=11)
    axes[1, 1].axis('off')

    # Mod 3: Hard negatives
    threshold_high = np.percentile(img_gray, 80)
    bright = (img_gray > threshold_high).astype(np.uint8) * 255
    non_lesion = cv2.bitwise_not(mask)
    hn_mask = cv2.bitwise_and(bright, non_lesion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hn_mask = cv2.morphologyEx(hn_mask, cv2.MORPH_OPEN, kernel)

    img_hn = img_np.copy()
    contours, _ = cv2.findContours(hn_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    hn_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 200 < area < 15000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_hn, (x, y), (x+w, y+h), (255, 50, 50), 2)
            hn_count += 1
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img_hn, (x1, y1), (x2, y2), (0, 255, 0), 3)

    axes[1, 2].imshow(img_hn)
    axes[1, 2].set_title(f'Mod 3: Hard Negatives (red)\nvs Lesions (green)',
                         fontsize=11, color='#c62828')
    axes[1, 2].axis('off')

    # Info box
    axes[1, 3].axis('off')
    info = (
        "How Each Modification Helps\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. Lesion-Aware Weighting\n"
        "   Dense loss × α on lesion regions\n"
        "   → Better lesion representations\n\n"
        "2. Asymmetric Augmentation\n"
        "   Global + local lesion-biased views\n"
        "   → Improved small object features\n\n"
        "3. Hard Negative Mining\n"
        "   Inject confusing normal regions\n"
        "   → More discriminative features\n\n"
        f"Stats for this image:\n"
        f"  Pseudo-lesions: {len(bboxes)}\n"
        f"  Hard neg candidates: {hn_count}"
    )
    axes[1, 3].text(0.05, 0.95, info, transform=axes[1, 3].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =========================================================================
# Main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Modified DenseCL Modifications"
    )
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing panoramic images')
    parser.add_argument('--output_dir', type=str, default='output/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if not HAS_MPL:
        print("Error: matplotlib is required. pip install matplotlib")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = sorted([
        os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if len(image_paths) == 0:
        print(f"No images found in {args.image_dir}")
        return

    random.seed(args.seed)
    selected = random.sample(image_paths, min(args.num_images, len(image_paths)))

    print(f"Visualizing {len(selected)} images from {args.image_dir}")
    print(f"Output: {args.output_dir}\n")

    for i, img_path in enumerate(selected):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(selected)}] {os.path.basename(img_path)}")

        # Mod 1: Lesion-Aware Weighting
        visualize_lesion_weighting(
            img_path,
            os.path.join(args.output_dir, f'{fname}_mod1_weighting.png')
        )

        # Mod 2: Asymmetric Augmentation
        visualize_asymmetric_augmentation(
            img_path,
            os.path.join(args.output_dir, f'{fname}_mod2_augmentation.png')
        )

        # Mod 3: Hard Negative Mining
        visualize_hard_negatives(
            img_path,
            os.path.join(args.output_dir, f'{fname}_mod3_hard_negatives.png')
        )

        # Combined overview
        visualize_pipeline_overview(
            img_path,
            os.path.join(args.output_dir, f'{fname}_overview.png')
        )

    print(f"\nDone! {len(selected) * 4} visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
