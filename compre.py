import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Set random seeds for reproducible visual sampling
random.seed(42)
np.random.seed(42)

# Hardware & Directory Setup
BASE_DIR = Path.cwd().resolve()
CUSTOM_MODEL_PATH = BASE_DIR / "saved_models" / "best.pt"
PRETRAINED_MODEL_PATH = BASE_DIR / "saved_models" / "helmet_detection_model.pt"
TEST_IMG_DIR = BASE_DIR / "HelmetDataset" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "HelmetDataset" / "test" / "labels"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def calculate_box_iou(box1, box2):
    """Calculates Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0

def evaluate_model_direct(model, img_dir, lbl_dir, label='Model', conf=0.25, iou_thresh=0.5):
    """Evaluates a YOLO model directly against ground-truth labels."""
    img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

    tp, fp, fn = 0, 0, 0
    class_stats = {
        0: {'name': 'With Helmet', 'tp': 0, 'fp': 0, 'fn': 0, 'gt_total': 0},
        1: {'name': 'Without Helmet', 'tp': 0, 'fp': 0, 'fn': 0, 'gt_total': 0}
    }
    matched_ious = []

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        h, w = img.shape[:2]

        lbl_p = lbl_dir / (p.stem + '.txt')
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    gt_box = [int((xc - bw/2)*w), int((yc - bh/2)*h), int((xc + bw/2)*w), int((yc + bh/2)*h)]
                    gt_boxes.append((cls_id, gt_box))
                    if cls_id in class_stats:
                        class_stats[cls_id]['gt_total'] += 1

        res = model.predict(img, conf=conf, device=DEVICE, verbose=False)[0]
        preds = []
        if res.boxes is not None:
            for b in res.boxes:
                preds.append((int(b.cls[0]), list(map(int, b.xyxy[0])), float(b.conf[0])))

        matched_gt = set()
        for pred_cls, pred_b, pred_c in preds:
            best_iou = 0.0
            best_gt = -1
            for i, (gt_cls, gt_b) in enumerate(gt_boxes):
                if i in matched_gt or pred_cls != gt_cls: continue
                iou = calculate_box_iou(pred_b, gt_b)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i

            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_gt)
                matched_ious.append(best_iou)
                if pred_cls in class_stats: class_stats[pred_cls]['tp'] += 1
            else:
                fp += 1
                if pred_cls in class_stats: class_stats[pred_cls]['fp'] += 1

        for i, (gt_cls, _) in enumerate(gt_boxes):
            if i not in matched_gt:
                fn += 1
                if gt_cls in class_stats: class_stats[gt_cls]['fn'] += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    for c_id in [0, 1]:
        c_tp, c_fp, c_fn = class_stats[c_id]['tp'], class_stats[c_id]['fp'], class_stats[c_id]['fn']
        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        class_stats[c_id]['f1'] = 2 * (c_p * c_r) / (c_p + c_r + 1e-6)

    return {'Model': label, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'Mean IoU': mean_iou,
            'TP': tp, 'FP': fp, 'FN': fn, 'class_stats': class_stats}

def main():
    print("=" * 65)
    print(f"🚀 MODEL COMPARISON STARTING ON DEVICE: {DEVICE_NAME}")
    print("=" * 65)

    if not CUSTOM_MODEL_PATH.exists() or not PRETRAINED_MODEL_PATH.exists():
        print("❌ Error: One or both model files are missing in saved_models/")
        return

    model_custom = YOLO(str(CUSTOM_MODEL_PATH))
    model_pretrained = YOLO(str(PRETRAINED_MODEL_PATH))

    print("🔍 Evaluating Model A (Custom)...")
    m_cust = evaluate_model_direct(model_custom, TEST_IMG_DIR, TEST_LBL_DIR, "Custom Trained")
    print("🔍 Evaluating Model B (Pretrained)...")
    m_pre = evaluate_model_direct(model_pretrained, TEST_IMG_DIR, TEST_LBL_DIR, "Pretrained")

    # Final Decision Scorecard
    score_a, score_b = 0, 0

    # Average Confidence Score Calculation
    def get_avg_conf(model, img_dir):
        img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        all_confs = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: continue
            res = model.predict(img, conf=0.25, device=DEVICE, verbose=False)[0]
            if res.boxes is not None:
                all_confs.extend(res.boxes.conf.tolist())
        return np.mean(all_confs) if all_confs else 0.0

    print("📈 Calculating average confidence scores...")
    conf_a = get_avg_conf(model_custom, TEST_IMG_DIR)
    conf_b = get_avg_conf(model_pretrained, TEST_IMG_DIR)

    winner = "Custom Trained" if conf_a > conf_b else "Pretrained"

    print("\n" + "=" * 65)
    print(f"🏆 FINAL VERDICT BASED ON CONFIDENCE: {winner.upper()} IS BETTER")
    print(f"Custom Model Avg Conf: {conf_a:.4f}")
    print(f"Pretrained Model Avg Conf: {conf_b:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
