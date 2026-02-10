from __future__ import annotations
import argparse
import json
import os
import sys
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.modifiers import (
    MODIFIER_REGISTRY,
    resolve_modifier_list
)
from src.shared.storage import DiskBackend, EphemeralBackend
from src.shared.inference import YOLOInference
from src.shared.utils import evaluate_map_at_iou
from src.shared.analysis import find_thresholds
from src.shared.pipeline import BenchmarkPipeline, DatasetLoader

DEFAULT_CONFIG = {
    "images_dir": None,
    "output_dir": "../outputs",
    "modifier": None,
    "passes": 10,
    "step_percent": 10.0,
    "storage_mode": "ram",
    "config_path": None,
    "plot": False
}

# CLI HELPERS

def check_storage_safety(n_images, n_passes, multiplier, avg_size_mb=1.5):
    # estimate: images * passes * modifiers * avg size
    total_mb = n_images * n_passes * multiplier * avg_size_mb
    total_gb = total_mb / 1024.0

    print(f"[Storage Check] Estimated space required: {total_gb:.2f} GB")

    # check disk space
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)

    if total_gb > free_gb:
        print(f"[!] CRITICAL: Not enough disk space! Available: {free_gb:.2f} GB")
        sys.exit(1)

    if total_gb > 10.0: # warn if over 10GB
        print(f"[!] WARNING: This run will consume significant storage ({total_gb:.2f} GB).")
        confirm = input("Are you sure you want to continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborting.")
            sys.exit(0)

def build_inference_backend():
    return YOLOInference()

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CV model robustness to image degradation")

    parser.add_argument("--images-dir", "-dir", type=str, help="Path to image folder")
    parser.add_argument("--modifier", "-mod", type=str, help="Which image degradation modifiers to use. Examples: 'resolution', ['luminance', 'contrast'], semantic, all")
    parser.add_argument("--passes", "-ps", type=int, help="Number of passes to simulate")
    parser.add_argument("--step-percent", "-st", type=float, help="Modification in percent per pass")
    parser.add_argument("--plot", "-pl", action="store_true", help="Show a plot of results")
    parser.add_argument("--storage-mode", "-sm", type=str, choices=["ram", "disk"], help="Storage mode: 'ram' (ephemeral) or 'disk'")
    parser.add_argument("--output-dir", "-out", type=str, help="Output directory for generated dataset")
    parser.add_argument("--config", "-c", dest="config_path", type=str, help="Path to JSON config file")
    
    return parser.parse_args()

def main_cli():
    cli_args = parse_args()

    # CONFIGURATION MERGE LOGIC (defaults < config < cli)

    final_config = DEFAULT_CONFIG.copy()

    # Load from Config File
    if cli_args.config_path and os.path.exists(cli_args.config_path):
        print(f"[Config] Loading defaults from {cli_args.config_path}")
        with open(cli_args.config_path, 'r') as f:
            json_config = json.load(f)
            # Only update keys that are not null in json
            clean_json = {k: v for k, v in json_config.items() if v is not None}
            final_config.update(clean_json)

    # Apply CLI Overrides
    for k, v in vars(cli_args).items():
        if k == "plot":
            if v is True: final_config[k] = True
        elif v is not None:
            final_config[k] = v

    if cli_args.config_path:
        final_config["config_path"] = cli_args.config_path

    args = argparse.Namespace(**final_config)

    # VALIDATION
    
    if not args.images_dir:
        print("[Error] --images-dir is required (via CLI or Config)")
        sys.exit(1)

    if not args.modifier:
        print("[Error] --modifier is required (via CLI or Config)")
        sys.exit(1)

    target_modifiers = resolve_modifier_list(args.modifier)
    if not target_modifiers:
        print(f"[Error] Could not resolve modifiers from: {args.modifier}")
        sys.exit(1)

    # SETUP
     
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") #generate unique run ID
    json_filename = f"thresholds_{run_id}.json"
    
    base_output_dir = args.output_dir
    
    if args.storage_mode == "disk":
        run_output_dir = os.path.join(base_output_dir, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"[Output] Created run directory: {run_output_dir}")
        
        storage = DiskBackend(run_output_dir)
        
        final_json_path = os.path.join(run_output_dir, json_filename)
        
        #Disk Space Safety Check
        if os.path.exists(args.images_dir):
            image_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            # Multiply estimate by number of modifiers
            check_storage_safety(len(image_files), args.passes, multiplier=len(target_modifiers))
        else:
            print(f"[Error] Images directory not found: {args.images_dir}")
            sys.exit(1)

    else:
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"[Output] RAM mode active. JSON will be saved to: {base_output_dir}")
        
        storage = EphemeralBackend()
        final_json_path = os.path.join(base_output_dir, json_filename)

    #init resources
    if not os.path.exists(args.images_dir):
        print(f"[Error] Images directory not found: {args.images_dir}")
        sys.exit(1)

    loader = DatasetLoader(args.images_dir)
    try:
        backend = build_inference_backend()
    except Exception as e:
        print(f"[Error] Inference backend failed: {e}")
        sys.exit(1)

    full_report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": {}
    }
    
    # MAIN LOOP

    for mod_name in target_modifiers:
        print(f"\n{'='*40}")
        print(f" PROCESSING: {mod_name.upper()}")
        print(f"{'='*40}")

        if mod_name not in MODIFIER_REGISTRY:
            print(f"[Warning] Skipping unknown modifier: {mod_name}")
            continue

        modifier_cls = MODIFIER_REGISTRY[mod_name]
        try:
            current_modifier = modifier_cls(step_percent=args.step_percent)
        except TypeError as e:
            print(f"[Error] Instantiation failed for {mod_name}: {e}")
            continue

        pipeline = BenchmarkPipeline(
            dataset_loader=loader,
            modifier=current_modifier,
            inference_backend=backend,
            storage_backend=storage,
            passes=args.passes,
            step_size_percent=args.step_percent
        )

        results = pipeline.run()
        
        # ANALYSIS

        baseline_entry = results.get(0)
        baseline_map = baseline_entry["score"] if baseline_entry else 0.0

        if baseline_map == 0.0:
            print("[Warning] Baseline mAP is 0.0. Check your model or dataset.")
        analysis_data = find_thresholds(results, baseline_map)
        
        #print summary
        if analysis_data.get("fail_pass"):
             print(f"[Analysis] Threshold reached at Pass {analysis_data['fail_pass']}")
             print(f"           Metrics: {analysis_data['thresholds']}")
        else:
             print(f"[Analysis] Robust (No threshold reached).")

        full_report["results"][mod_name] = {
            "analysis": analysis_data,
            "raw_scores": results
        }
    
    #save final json
    try:
        with open(final_json_path, "w") as f:
            json.dump(full_report, f, indent=4)
        print(f"\n[Success] Full analysis saved to: {final_json_path}")
    except Exception as e:
        print(f"[Error] Failed to save JSON: {e}")

if __name__ == "__main__":
    main_cli()