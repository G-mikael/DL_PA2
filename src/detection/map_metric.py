import numpy as np
from typing import List, Dict
from src.metrics.tracker_metrics import compute_iou

def compute_frame_ap(gt_boxes: List[dict], pred_boxes: List[dict], iou_threshold: float = 0.5) -> float:
    """Calcula a Average Precision (AP) para um único quadro."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Ordena predições por confiança decrescente
    preds_sorted = sorted(pred_boxes, key=lambda x: x.get('conf', 1.0), reverse=True)
    
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    gt_matched = set()

    for i, p in enumerate(preds_sorted):
        box_p = (p['bb_left'], p['bb_top'], p['bb_width'], p['bb_height'])
        best_iou = iou_threshold
        best_gt_idx = -1

        for j, g in enumerate(gt_boxes):
            if j in gt_matched:
                continue
            box_g = (g['bb_left'], g['bb_top'], g['bb_width'], g['bb_height'])
            iou = compute_iou(box_p, box_g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_gt_idx >= 0:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

    # Cálculo do AP pela área sob a curva Precision-Recall
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for k in range(len(precisions) - 1, 0, -1):
        precisions[k - 1] = np.maximum(precisions[k - 1], precisions[k])
    
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)