"""
Evaluation Metrics for IO Detection
======================================
COCO-style mAP, IoU, Precision/Recall computation.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes.
    
    Args:
        box1 (Tensor or array): (4,) in [x1, y1, x2, y2] format.
        box2 (Tensor or array): (4,) in [x1, y1, x2, y2] format.
    
    Returns:
        float: Intersection over Union.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(boxes1, boxes2):
    """Compute pairwise IoU matrix between two sets of boxes.
    
    Args:
        boxes1 (Tensor): (N, 4) predicted boxes.
        boxes2 (Tensor): (M, 4) ground truth boxes.
    
    Returns:
        Tensor: (N, M) IoU matrix.
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1).float()
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2).float()
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    if N == 0 or M == 0:
        return torch.zeros(N, M)
    
    # Expand for broadcasting
    b1 = boxes1.unsqueeze(1).expand(N, M, 4)  # (N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)  # (N, M, 4)
    
    # Intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h
    
    # Areas
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    
    union = area1 + area2 - intersection
    
    iou = intersection / union.clamp(min=1e-6)
    
    return iou


def compute_ap(precision, recall):
    """Compute Average Precision using 11-point interpolation (PASCAL VOC style).
    
    Args:
        precision (np.ndarray): Precision values at each detection.
        recall (np.ndarray): Recall values at each detection.
    
    Returns:
        float: Average precision.
    """
    # Add sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum ΔRecall × Precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_ap_coco(precision, recall):
    """Compute Average Precision using 101-point interpolation (COCO style).
    
    Args:
        precision (np.ndarray): Precision values.
        recall (np.ndarray): Recall values.
    
    Returns:
        float: COCO-style AP.
    """
    recall_thresholds = np.linspace(0, 1, 101)
    precisions = np.zeros(101)
    
    for i, t in enumerate(recall_thresholds):
        mask = recall >= t
        if mask.any():
            precisions[i] = precision[mask].max()
    
    return precisions.mean()


def evaluate_detection(predictions, ground_truths, iou_thresholds=None):
    """Evaluate object detection results using COCO-style metrics.
    
    Args:
        predictions (list): List of dicts, one per image:
            {'boxes': (N, 4), 'scores': (N,), 'labels': (N,)}
        ground_truths (list): List of dicts, one per image:
            {'boxes': (M, 4), 'labels': (M,)}
        iou_thresholds (list): IoU thresholds for evaluation.
            Default: [0.5] for AP50, or [0.5:0.95:0.05] for COCO mAP.
    
    Returns:
        dict: Evaluation results with keys:
            'mAP': COCO-style mAP (average over IoU thresholds)
            'AP50': AP at IoU=0.5
            'AP75': AP at IoU=0.75
            'precision': precision array
            'recall': recall array
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    results = {}
    aps_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        # Collect all predictions and matches across images
        all_scores = []
        all_tp = []
        total_gt = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = gt['boxes']
            
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            
            num_gt = len(gt_boxes)
            total_gt += num_gt
            
            if len(pred_boxes) == 0:
                continue
            
            if num_gt == 0:
                all_scores.extend(pred_scores.tolist())
                all_tp.extend([0] * len(pred_scores))
                continue
            
            # Sort predictions by score (descending)
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            
            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(
                torch.from_numpy(pred_boxes), 
                torch.from_numpy(gt_boxes)
            ).numpy()
            
            # Match predictions to ground truths (greedy)
            gt_matched = np.zeros(num_gt, dtype=bool)
            
            for i in range(len(pred_boxes)):
                all_scores.append(pred_scores[i])
                
                if num_gt == 0:
                    all_tp.append(0)
                    continue
                
                # Find best matching GT
                best_gt = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_gt]
                
                if best_iou >= iou_thresh and not gt_matched[best_gt]:
                    all_tp.append(1)
                    gt_matched[best_gt] = True
                else:
                    all_tp.append(0)
        
        if total_gt == 0:
            aps_per_threshold.append(0.0)
            continue
        
        # Compute precision and recall
        all_scores = np.array(all_scores)
        all_tp = np.array(all_tp)
        
        # Sort by score (descending)
        sort_idx = np.argsort(-all_scores)
        all_tp = all_tp[sort_idx]
        
        cum_tp = np.cumsum(all_tp)
        cum_fp = np.cumsum(1 - all_tp)
        
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / total_gt
        
        # Compute AP
        ap = compute_ap_coco(precision, recall)
        aps_per_threshold.append(ap)
        
        # Store detailed results for key thresholds
        if abs(iou_thresh - 0.5) < 1e-6:
            results['AP50'] = ap
            results['precision_50'] = precision
            results['recall_50'] = recall
        if abs(iou_thresh - 0.75) < 1e-6:
            results['AP75'] = ap
    
    # COCO mAP = average AP over all IoU thresholds
    results['mAP'] = np.mean(aps_per_threshold)
    results['APs_per_threshold'] = dict(zip(
        [f"AP{int(t*100)}" for t in iou_thresholds], 
        aps_per_threshold
    ))
    
    return results


def print_eval_results(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 50)
    print("Detection Evaluation Results")
    print("=" * 50)
    print(f"  mAP (COCO):  {results['mAP']:.4f}")
    if 'AP50' in results:
        print(f"  AP50:        {results['AP50']:.4f}")
    if 'AP75' in results:
        print(f"  AP75:        {results['AP75']:.4f}")
    
    if 'APs_per_threshold' in results:
        print("\n  Per-threshold APs:")
        for k, v in results['APs_per_threshold'].items():
            print(f"    {k}: {v:.4f}")
    print("=" * 50)
