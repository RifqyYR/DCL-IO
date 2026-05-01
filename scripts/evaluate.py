"""
Evaluation Script for IO Detection
=====================================
Evaluates Faster R-CNN on test data with comprehensive metrics:
  - Sensitivity (Recall) at IoU=0.5
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - Inference time (per image, FPS)

Usage:
    python scripts/evaluate.py \
        --image_dir data/test/images \
        --annotation_file data/test/annotations.json \
        --checkpoint output/finetune/v3/checkpoints/best.pth
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np
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
                        help='Min detection score')
    parser.add_argument('--warmup_iters', type=int, default=10,
                        help='GPU warmup iterations before timing')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
        dummy = torch.randn(1, 3, 512, 512, device=device)
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


if __name__ == '__main__':
    main()
