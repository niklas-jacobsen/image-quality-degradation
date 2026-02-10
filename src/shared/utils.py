from typing import List, Dict, Tuple, Any, Optional
import numpy as np

def iou_xyxy(a: List[float], b: List[float]) -> float:
    """
    computes IoU between two boxes in [x1,y1,x2,y2] format
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def evaluate_map_at_iou(preds: List[Any], gts: List[List[Dict]], iou_thresh: float = 0.5) -> float:
    """
    small mAP@IoU implementation for all classes, averaged over classes, at a single IoU threshold.
    preds, gts are aligned lists over samples.
    gts per sample is a list of dicts with keys 'bbox' [x1,y1,x2,y2], 'category_id'
    """
    from typing import Any

    #collect detections by class
    per_class_detections: Dict[int, List[Tuple[float,int,int]]] = {}  # class -> list of (score, sample_idx, det_idx)
    per_class_gt_counts: Dict[int, int] = {}
    detection_lookup: Dict[Tuple[int,int], Dict] = {}  # (sample_idx, det_idx) -> predicted box
    for si, pred in enumerate(preds):
        for di, (box, score, label) in enumerate(zip(pred.boxes, pred.scores, pred.labels)):
            per_class_detections.setdefault(label, []).append((score, si, di))
            detection_lookup[(si, di)] = {"box": box, "used": False}
    for si, gt_list in enumerate(gts):
        for g in gt_list:
            cid = g["category_id"]
            per_class_gt_counts[cid] = per_class_gt_counts.get(cid, 0) + 1

    aps = []
    for cid, dets in per_class_detections.items():
        # sort by score descending
        dets_sorted = sorted(dets, key=lambda x: x[0], reverse=True)
        tp = []
        fp = []
        matched_gt = {}  # sample_idx -> list of matched gt indices
        total_gt = per_class_gt_counts.get(cid, 0)
        for score, si, di in dets_sorted:
            pred_box = detection_lookup[(si, di)]["box"]
            # find best matching gt in that sample of same class
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gts[si]):
                if g["category_id"] != cid:
                    continue
                iouv = iou_xyxy(pred_box, g["bbox"])
                if iouv > best_iou:
                    best_iou = iouv
                    best_gi = gi
            if best_iou >= iou_thresh and (si, best_gi) not in matched_gt:
                tp.append(1)
                fp.append(0)
                matched_gt[(si, best_gi)] = True
            else:
                tp.append(0)
                fp.append(1)
        if total_gt == 0:
            # no ground truth for this class, skip from averaging
            continue
        # compute precision recall arrays
        tp_cum = np.cumsum(tp).astype(float)
        fp_cum = np.cumsum(fp).astype(float)
        recalls = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        #compute AP
        if precisions.size == 0:
            ap = 0.0
        else:
            #ensure monotonic precision
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            #integrate
            ap = 0.0
            recall_points = np.concatenate(([0.0], recalls, [1.0]))
            prec_points = np.concatenate(([0.0], precisions, [0.0]))
            ap = np.trapezoid(prec_points, recall_points)
        aps.append(ap)
    if not aps:
        return 0.0
    return float(np.mean(aps))
