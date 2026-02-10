import numpy as np

def calculate_kneedle(values):
    """Finds the 'elbow' (maximum curvature) of the curve."""
    y = np.array(values)
    x = np.arange(len(y))
    
    if len(y) < 3 or (y.max() - y.min()) == 0: return None
    
    y_norm = (y - y.min()) / (y.max() - y.min())
    x_norm = (x - x.min()) / (x.max() - x.min())
    
    start_point = np.array([x_norm[0], y_norm[0]])
    end_point = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end_point - start_point
    
    distances = []
    for i in range(len(y)):
        point = np.array([x_norm[i], y_norm[i]])
        vec_from_start = point - start_point
        proj = np.dot(vec_from_start, line_vec) / np.dot(line_vec, line_vec)
        proj_point = start_point + proj * line_vec
        distances.append(np.linalg.norm(point - proj_point))
        
    return np.argmax(distances)

def analyze_metric(passes, values, drop_thresh, floor_thresh=None):
    """
    Calculates all 3 threshold types for a single metric series.
    Returns a dict with the 3 pass candidates.
    """
    baseline = values[0]
    
    #relative drop
    drop_pass = None
    if baseline > 0:
        limit = baseline * (1.0 - drop_thresh)
        for i, v in enumerate(values):
            if v < limit:
                drop_pass = passes[i]
                break

    #safety floor
    floor_pass = None
    if floor_thresh is not None:
        for i, v in enumerate(values):
            if v < floor_thresh:
                floor_pass = passes[i]
                break
                
    #kneedle
    knee_idx = calculate_kneedle(values)
    knee_pass = passes[knee_idx] if knee_idx is not None else None
    
    return {
        "drop": drop_pass,
        "kneedle": knee_pass,
        "floor": floor_pass
    }

def find_thresholds(results_dict: dict, baseline_score: float) -> dict:
    passes = sorted(results_dict.keys())
    
    #extract series
    series = {
        "map": [results_dict[p]["score"] for p in passes],
        "conf": [results_dict[p].get("avg_confidence", 0) for p in passes],
        "count": [results_dict[p].get("avg_detection_count", 0) for p in passes]
    }

    #analyze metrics: map, confidence, and count
    map_res = analyze_metric(passes, series["map"], drop_thresh=0.40, floor_thresh=0.25)
    conf_res = analyze_metric(passes, series["conf"], drop_thresh=0.20, floor_thresh=0.50)
    count_res = analyze_metric(passes, series["count"], drop_thresh=0.50, floor_thresh=None)

    #determine pass with earliest failure
    all_candidates = []
    for res in [map_res, conf_res, count_res]:
        for val in res.values():
            if val is not None:
                all_candidates.append(val)

    if not all_candidates:
        return {"status": "robust", "fail_pass": None, "thresholds": None}

    #find earliest failure
    failure_pass = min(all_candidates)
    
    failure_metrics = results_dict[failure_pass]["metrics"]
    baseline_metrics = results_dict[0]["metrics"]

    final_thresholds = {}
    
    #metrics to export
    keys = ["brightness", "contrast", "sharpness", "entropy", "overexposure"]
    
    for k in keys:
        fail_val = failure_metrics.get(k)
        base_val = baseline_metrics.get(k)
        
        if fail_val is None or base_val is None:
            continue
            
        #determine direction constraint
        if fail_val > base_val:
            direction = "<" 
        else:
            direction = ">"
            
        final_thresholds[k] = {
            "value": fail_val,
            "direction_constraint": direction 
        }

    return {
        "status": "threshold_found",
        "fail_pass": failure_pass,
        "thresholds": final_thresholds,
        "matrix": {
            "map": map_res,
            "confidence": conf_res,
            "count": count_res
        }
    }