"""
src/apply_thresholds.py

Phase III: Validation / Application Script.
Validated version with corrected physics logic.
"""

import argparse
import json
import os
import sys
import cv2 as cv
import numpy as np
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.metrics import measure_all

def load_all_configs(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        raise ValueError("JSON format error: 'results' key missing")

    configs = []
    
    for mod_name, mod_data in results.items():
        analysis = mod_data.get("analysis", {})
        if analysis.get("status") != "threshold_found":
            print(f"[Skip] Modifier '{mod_name}' did not find a stable threshold. Skipping.")
            continue
            
        thresholds = analysis.get("thresholds")
        if not thresholds:
            continue

        raw_scores = mod_data.get("raw_scores", {})
        baseline_metrics = raw_scores.get("0", {}).get("metrics", {})
        
        if not baseline_metrics:
            print(f"[Warning] '{mod_name}' has no baseline metrics. Assuming all thresholds relevant.")
            baseline_metrics = thresholds 

        configs.append({
            "name": mod_name,
            "thresholds": thresholds,
            "baseline": baseline_metrics
        })
        
    print(f"[Config] Loaded profiles for {len(configs)} modifiers: {[c['name'] for c in configs]}")
    return configs

def get_active_constraints(configs: List[Dict]) -> List[Dict]:
    constraints = []
    
    print("\n[Init] Building Master Constraint List:")
    print(f"{'Source':<15} | {'Metric':<12} | {'Limit':<10} | {'Constraint'} | {'Status'}")
    print("-" * 85)

    for conf in configs:
        thresholds = conf["thresholds"]
        baseline = conf["baseline"]
        source = conf["name"]
        
        for key, data in thresholds.items():
            #parse Limit
            if isinstance(data, dict) and "value" in data:
                limit = float(data["value"])
                direction = data["direction_constraint"]
            else:
                limit = float(data)
                #legacy fallback
                if key == "overexposure": direction = "<" 
                elif key == "sharpness" and limit > 1000: direction = "<"
                else: direction = ">"
            
            #skip invalid limits
            if limit < 0.0001:
                continue

            #check if threshold is relevant compared to baseline
            base = baseline.get(key, 0)
            if base == 0: change = 1.0
            else: change = abs(limit - base) / base

            is_relevant = False
            if key == "overexposure":
                if limit > 0.02: is_relevant = True
            elif change > 0.10: 
                is_relevant = True
            
            if not is_relevant:
                continue

            # LOGICAL GUARDRAILS
            status = "ACTIVE"
            
            #handle ceiling killers
            if direction == "<":
                if key in ["brightness", "contrast", "entropy"]:
                    status = f"SKIPPED (Illogical: {key.capitalize()} Ceiling)"

            #handle floor killers
            if direction == ">":
                if key == "overexposure":
                    status = "SKIPPED (Illogical: Overexposure Floor)"

            visual_rule = f"Img {direction} {limit:.2f}"
            print(f"{source:<15} | {key:<12} | {limit:<10.2f} | {visual_rule:<12} | {status}")

            if status == "ACTIVE":
                constraints.append({
                    "metric": key,
                    "limit": limit,
                    "direction": direction,
                    "source": source
                })

    return constraints

def check_image(img_metrics: Dict, constraints: List[Dict]) -> Tuple[bool, List[str]]:
    failed = False
    reasons = []

    for c in constraints:
        metric = c["metric"]
        limit = c["limit"]
        direction = c["direction"]
        val = img_metrics.get(metric, 0)
        
        # direction "<" : safe if Low. fail if val >= limit
        if direction == "<":
            if val >= limit:
                failed = True
                reasons.append(f"{metric} too high ({val:.4f} >= {limit:.4f}) [{c['source']}]")
                
        # direction ">" : safe if High. fail if val <= limit
        elif direction == ">":
            if val <= limit:
                failed = True
                reasons.append(f"{metric} too low ({val:.4f} <= {limit:.4f}) [{c['source']}]")

    return not failed, reasons

def main():
    parser = argparse.ArgumentParser(description="Filter images using MULTIPLE quality thresholds")
    parser.add_argument("--images-dir", "-dir", required=True, help="Path to input images")
    parser.add_argument("--thresholds-json", "-json", required=True, help="Path to generated thresholds.json")
    parser.add_argument("--output-json", "-out", default="validation_report.json", help="Path to save report")
    parser.add_argument("--move-images-to", "-move", default=None, help="Optional: Move failed images here")
    
    args = parser.parse_args()

    try:
        configs = load_all_configs(args.thresholds_json)
        if not configs:
            print("[Error] No valid modifier profiles found in JSON.")
            sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    constraints = get_active_constraints(configs)
    
    if not constraints:
        print("[Warning] No active constraints generated. All images will pass.")
    
    if args.move_images_to:
        os.makedirs(args.move_images_to, exist_ok=True)

    valid_exts = ('.jpg', '.png', '.jpeg')
    files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(valid_exts)]
    
    stats = {"total": 0, "passed": 0, "failed": 0}
    report_details = {}

    print(f"\n[Run] checking {len(files)} images...")

    for i, fname in enumerate(files):
        fpath = os.path.join(args.images_dir, fname)
        img = cv.imread(fpath)
        
        if img is None:
            continue

        metrics = measure_all(img)
        passed, reasons = check_image(metrics, constraints)
        
        stats["total"] += 1
        if passed:
            stats["passed"] += 1
            status = "PASS"
        else:
            stats["failed"] += 1
            status = "FAIL"
            if args.move_images_to:
                fail_path = os.path.join(args.move_images_to, f"FAIL_{fname}")
                cv.imwrite(fail_path, img)

        report_details[fname] = {
            "status": status,
            "reasons": reasons,
            "metrics": metrics
        }
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(files)} (Failed: {stats['failed']})", end="\r")

    print(f"\n\n[Complete]")
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    
    output_path = args.output_json
    
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "validation_report.json")
    else:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({"summary": stats, "thresholds_used": {"file": args.thresholds_json, "constraints": constraints}, "details": report_details}, f, indent=4)
    
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()

    