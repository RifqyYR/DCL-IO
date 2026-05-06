"""
Evaluation Script for IO Detection
=====================================
Evaluates Faster R-CNN on test data with comprehensive metrics:
  - Sensitivity (Recall) at IoU=0.5
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - Inference time (per image, FPS)
  - Visualization of predictions with bounding boxes

Usage:
    python scripts/evaluate.py \
        --image_dir data/test/images \
        --annotation_file data/test/annotations.json \
        --checkpoint output/finetune/v3/checkpoints/best.pth \
        --vis_count 20
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_finetuning import DentalPanoramicDetectionDataset
from models.faster_rcnn import build_faster_rcnn


# =========================================================================
# Core metric functions
# =========================================================================

def compute_iou_matrix(boxes1, boxes2):
    """Pairwise IoU between two sets of boxes (N,4) and (M,4)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1 = boxes1[:, None, :]  # (N,1,4)
    b2 = boxes2[None, :, :]  # (1,M,4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def compute_ap_101(precision, recall):
    """COCO-style 101-point interpolated AP."""
    recall_thresholds = np.linspace(0, 1, 101)
    precs = np.zeros(101)
    for i, t in enumerate(recall_thresholds):
        mask = recall >= t
        if mask.any():
            precs[i] = precision[mask].max()
    return precs.mean()


def evaluate_at_iou(predictions, ground_truths, iou_thresh):
    """Evaluate at a single IoU threshold.

    Returns: (AP, precision_array, recall_array, sensitivity, specificity_info)
    """
    all_scores = []
    all_tp = []
    total_gt = 0

    # Per-image sensitivity tracking
    images_with_gt = 0
    images_detected = 0  # images where at least 1 GT was matched

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        num_gt = len(gt_boxes)
        total_gt += num_gt

        if num_gt > 0:
            images_with_gt += 1

        if len(pred_boxes) == 0:
            continue

        if num_gt == 0:
            all_scores.extend(pred_scores.tolist())
            all_tp.extend([0] * len(pred_scores))
            continue

        # Sort by score descending
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        gt_matched = np.zeros(num_gt, dtype=bool)

        img_has_match = False
        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i])
            best_gt = np.argmax(iou_mat[i])
            best_iou = iou_mat[i, best_gt]

            if best_iou >= iou_thresh and not gt_matched[best_gt]:
                all_tp.append(1)
                gt_matched[best_gt] = True
                img_has_match = True
            else:
                all_tp.append(0)

        if img_has_match:
            images_detected += 1

    if total_gt == 0:
        return 0.0, np.array([]), np.array([]), 0.0

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)

    order = np.argsort(-all_scores)
    all_tp = all_tp[order]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)

    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / total_gt

    ap = compute_ap_101(precision, recall)

    # Sensitivity = max recall achievable (i.e. recall at last detection)
    sensitivity = recall[-1] if len(recall) > 0 else 0.0

    return ap, precision, recall, sensitivity


# =========================================================================
# Main evaluation
# =========================================================================

def detection_collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IO Detection on Test Set")
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='COCO-format annotation JSON for test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint (.pth)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save results to JSON')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (use 1 for accurate inference time)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--score_thresh', type=float, default=0.05,
                        help='Min detection score for metrics')
    parser.add_argument('--warmup_iters', type=int, default=10,
                        help='GPU warmup iterations before timing')
    
    # Visualization
    parser.add_argument('--vis_count', type=int, default=20,
                        help='Number of sample images to visualize (0=disable)')
    parser.add_argument('--vis_score_thresh', type=float, default=0.3,
                        help='Min score for drawing prediction boxes')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='Directory to save visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Default visualization directory
    vis_dir = args.vis_dir
    if vis_dir is None:
        vis_dir = os.path.join(os.path.dirname(args.checkpoint), 'visualizations')

    # --- Load dataset ---
    dataset = DentalPanoramicDetectionDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        transforms=None,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
    )

    # --- Load model ---
    model = build_faster_rcnn(num_classes=args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if 'epoch' in ckpt:
        print(f"  Epoch: {ckpt['epoch'] + 1}")
    if 'mAP' in ckpt:
        print(f"  Checkpoint mAP: {ckpt['mAP']:.4f}")

    # --- GPU warmup (for accurate timing) ---
    if device.type == 'cuda':
        print(f"\nWarming up GPU ({args.warmup_iters} iters)...")
        dummy = torch.randn(3, 512, 512, device=device)
        for _ in range(args.warmup_iters):
            with torch.no_grad():
                _ = model([dummy])
        torch.cuda.synchronize()

    # --- Inference ---
    print(f"\nRunning inference on {len(dataset)} images...")
    all_predictions = []
    all_ground_truths = []
    inference_times = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inference"):
            images_gpu = [img.to(device) for img in images]

            # Time inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            outputs = model(images_gpu)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            inference_times.append((t_end - t_start) / len(images))

            for output, target in zip(outputs, targets):
                keep = output['scores'] >= args.score_thresh
                all_predictions.append({
                    'boxes': output['boxes'][keep].cpu().numpy(),
                    'scores': output['scores'][keep].cpu().numpy(),
                    'labels': output['labels'][keep].cpu().numpy(),
                })
                all_ground_truths.append({
                    'boxes': target['boxes'].numpy(),
                    'labels': target['labels'].numpy(),
                })

    # --- Compute metrics at multiple IoU thresholds ---
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95
    ap_per_iou = {}

    for iou_t in iou_thresholds:
        key = f"AP{int(iou_t * 100)}"
        ap, prec, rec, sens = evaluate_at_iou(
            all_predictions, all_ground_truths, iou_t
        )
        ap_per_iou[key] = ap

    # Key metrics
    ap50 = ap_per_iou['AP50']
    ap75 = ap_per_iou['AP75']
    map_50_95 = np.mean(list(ap_per_iou.values()))

    # Sensitivity at IoU=0.5
    _, _, _, sensitivity = evaluate_at_iou(
        all_predictions, all_ground_truths, 0.5
    )

    # Inference time
    times = np.array(inference_times)
    avg_time_ms = times.mean() * 1000
    std_time_ms = times.std() * 1000
    fps = 1.0 / times.mean() if times.mean() > 0 else 0

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — IO Detection (Test Set)")
    print("=" * 60)

    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Sensitivity (Recall@IoU=0.5)':<30} {sensitivity:>10.4f}")
    print(f"  {'mAP@0.5':<30} {ap50:>10.4f}")
    print(f"  {'mAP@0.75':<30} {ap75:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<30} {map_50_95:>10.4f}")
    print(f"  {'Inference Time (ms/image)':<30} {avg_time_ms:>10.2f}")
    print(f"  {'FPS':<30} {fps:>10.1f}")

    print(f"\n  Per-IoU AP breakdown:")
    for key in sorted(ap_per_iou.keys(), key=lambda x: int(x[2:])):
        print(f"    {key}: {ap_per_iou[key]:.4f}")

    print(f"\n  Inference stats:")
    print(f"    Mean: {avg_time_ms:.2f} ms ± {std_time_ms:.2f} ms")
    print(f"    Total images: {len(dataset)}")
    print(f"    Device: {device}")
    print("=" * 60)

    # --- Save results ---
    results = {
        'sensitivity': float(sensitivity),
        'mAP_50': float(ap50),
        'mAP_75': float(ap75),
        'mAP_50_95': float(map_50_95),
        'inference_time_ms': float(avg_time_ms),
        'inference_time_std_ms': float(std_time_ms),
        'fps': float(fps),
        'per_iou_ap': {k: float(v) for k, v in ap_per_iou.items()},
        'num_images': len(dataset),
        'device': str(device),
        'checkpoint': args.checkpoint,
        'score_threshold': args.score_thresh,
    }

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(args.checkpoint), 'test_results.json'
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # --- Visualize predictions ---
    if args.vis_count > 0:
        visualize_predictions(
            dataset=dataset,
            model=model,
            device=device,
            num_images=args.vis_count,
            score_thresh=args.vis_score_thresh,
            output_dir=vis_dir,
        )


# =========================================================================
# Visualization
# =========================================================================

def visualize_predictions(dataset, model, device, num_images, score_thresh,
                          output_dir):
    """Draw GT and prediction bounding boxes on sample images and save.
    
    - Green boxes = Ground Truth
    - Red boxes = Predictions (with confidence score)
    - Blue dashed = False Negatives (GT not matched)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Select images: prefer images that have annotations
    indices_with_gt = []
    indices_without_gt = []
    for i in range(len(dataset)):
        img_id = dataset.image_ids[i]
        anns = dataset.annotations.get(img_id, [])
        if len(anns) > 0:
            indices_with_gt.append(i)
        else:
            indices_without_gt.append(i)
    
    # Mix: prioritize images with GT, fill rest with no-GT
    np.random.seed(42)
    selected = []
    if len(indices_with_gt) > 0:
        n_gt = min(num_images, len(indices_with_gt))
        selected.extend(np.random.choice(indices_with_gt, n_gt, replace=False).tolist())
    remaining = num_images - len(selected)
    if remaining > 0 and len(indices_without_gt) > 0:
        n_no = min(remaining, len(indices_without_gt))
        selected.extend(np.random.choice(indices_without_gt, n_no, replace=False).tolist())
    
    print(f"\nVisualizing {len(selected)} images → {output_dir}")
    
    for idx in tqdm(selected, desc="Drawing"):
        image_tensor, target = dataset[idx]
        img_id = dataset.image_ids[idx]
        img_info = dataset.images[img_id]
        filename = img_info['file_name']
        
        # Load original image (full resolution, not tensor)
        img_path = os.path.join(str(dataset.image_dir), filename)
        pil_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_image.size
        
        # Run inference on tensor
        with torch.no_grad():
            outputs = model([image_tensor.to(device)])
        output = outputs[0]
        
        # Tensor dimensions (may differ from original if dataset resized)
        _, tensor_h, tensor_w = image_tensor.shape
        scale_x = orig_w / tensor_w
        scale_y = orig_h / tensor_h
        
        # Prepare drawing
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, orig_h // 60))
            font_small = ImageFont.truetype("arial.ttf", size=max(11, orig_h // 80))
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font
        
        line_width = max(2, orig_h // 300)
        
        # --- Draw Ground Truth boxes (green) ---
        gt_boxes = target['boxes'].numpy()
        gt_matched = set()
        
        # Check which GTs are matched by predictions (IoU >= 0.5)
        pred_boxes_np = output['boxes'].cpu().numpy()
        pred_scores_np = output['scores'].cpu().numpy()
        keep = pred_scores_np >= score_thresh
        pred_boxes_filtered = pred_boxes_np[keep]
        pred_scores_filtered = pred_scores_np[keep]
        
        if len(gt_boxes) > 0 and len(pred_boxes_filtered) > 0:
            # Scale pred boxes to original image coords
            pred_scaled = pred_boxes_filtered.copy()
            pred_scaled[:, [0, 2]] *= scale_x
            pred_scaled[:, [1, 3]] *= scale_y
            
            gt_scaled = gt_boxes.copy()
            gt_scaled[:, [0, 2]] *= scale_x
            gt_scaled[:, [1, 3]] *= scale_y
            
            iou_mat = compute_iou_matrix(pred_scaled, gt_scaled)
            for gi in range(len(gt_scaled)):
                max_iou = iou_mat[:, gi].max() if len(iou_mat) > 0 else 0
                if max_iou >= 0.5:
                    gt_matched.add(gi)
        
        for gi, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            # Scale to original image
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            if gi in gt_matched:
                # Matched GT → solid green
                color = (0, 200, 0)
                label = "GT (matched)"
            else:
                # Unmatched GT (false negative) → blue
                color = (0, 100, 255)
                label = "GT (missed)"
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            # Label background
            text_bbox = draw.textbbox((x1, y1 - 18), label, font=font_small)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font_small)
        
        # --- Draw Prediction boxes (red) ---
        for pi in range(len(pred_boxes_filtered)):
            box = pred_boxes_filtered[pi]
            score = pred_scores_filtered[pi]
            
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            # Color by confidence: bright red for high, dim for low
            intensity = int(150 + 105 * score)
            color = (intensity, 30, 30)
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            label = f"IO {score:.2f}"
            text_bbox = draw.textbbox((x1, y2 + 2), label, font=font_small)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y2 + 2), label, fill=(255, 255, 255), font=font_small)
        
        # --- Add summary text ---
        summary = (f"GT: {len(gt_boxes)} | "
                   f"Pred: {len(pred_boxes_filtered)} | "
                   f"Matched: {len(gt_matched)}")
        text_bbox = draw.textbbox((10, 10), summary, font=font)
        draw.rectangle(
            [text_bbox[0]-4, text_bbox[1]-4, text_bbox[2]+4, text_bbox[3]+4],
            fill=(0, 0, 0, 180)
        )
        draw.text((10, 10), summary, fill=(255, 255, 255), font=font)
        
        # Save
        save_name = f"vis_{os.path.splitext(filename)[0]}.jpg"
        save_path = os.path.join(output_dir, save_name)
        pil_image.save(save_path, quality=95)
    
    print(f"Saved {len(selected)} visualizations to {output_dir}")


if __name__ == '__main__':
    main()
