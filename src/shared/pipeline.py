from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
import cv2 as cv
import os
import time
import json
import random
from dataclasses import dataclass
from collections import deque

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future

from .inference import InferenceBackend, InferenceResult
from .storage import StorageBackend
from .modifiers import ModifierStrategy
from .utils import evaluate_map_at_iou
from .metrics import measure_all

@dataclass
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int

def generate_batch_task(base_img: np.ndarray, rec: ImageRecord, pass_range: range, modifier: ModifierStrategy) -> List[Tuple[int, np.ndarray]]:
    """
    Generates modified images for all passes.
    Designed to run in a separate process to avoid GIL and global random state interference.
    """
    batch_data = []
    for p in pass_range:
        if p == 0:
            mod_img = base_img.copy()
        else:
            if hasattr(modifier, 'apply'):
                # deterministic seeding based on image ID and pass
                seed_val = 42 + p + rec.id
                random.seed(seed_val)
                np.random.seed(seed_val)
                mod_img = modifier.apply(base_img, p)
            else:
                mod_img = base_img.copy()
        batch_data.append((p, mod_img))
    return batch_data

class DatasetLoader:
    """
    Loads images from disk.
    """

    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self.images: Dict[int, ImageRecord] = {}

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        for idx, fname in enumerate(sorted(os.listdir(images_dir))):
            path = os.path.join(images_dir, fname)

            if not os.path.isfile(path):
                continue

            if not fname.lower().endswith(valid_exts):
                continue

            self.images[idx] = ImageRecord(
                id=idx,
                file_name=fname,
                width=0,
                height=0
            )

    def iter_samples(self) -> Iterable[Tuple[str, np.ndarray, ImageRecord]]:
        """
        Iterate over (filepath, image numpy, ImageRecord)
        """
        for rec in self.images.values():
            path = os.path.join(self.images_dir, rec.file_name)
            if not os.path.exists(path):
                continue
            img = cv.imread(path, cv.IMREAD_COLOR)
            if img is None:
                continue
            yield path, img, rec

def _baseline_key(rec: ImageRecord, path: str) -> str:
    return rec.file_name if rec.file_name else path

def _load_image_safe(rec: ImageRecord, images_dir: str) -> Optional[Tuple[str, np.ndarray, ImageRecord]]:
    path = os.path.join(images_dir, rec.file_name)
    if not os.path.exists(path):
        return None
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        return None
    return path, img, rec

def compute_baseline(dataset_loader: DatasetLoader, inference_backend: InferenceBackend, batch_size: int = 32) -> Dict[str, List[Dict]]:
    """
    Computes baseline predictions for the entire dataset using parallel loading.
    """
    baseline_ground_truth = {}
    
    records = list(dataset_loader.images.values())
    total_images = len(records)
    print(f"[compute_baseline] Generating baseline ground truth for {total_images} images (Batch Size: {batch_size})...")

    num_workers = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=num_workers) as loader_executor:
        for i in range(0, total_images, batch_size):
            chunk_records = records[i : i + batch_size]
            
            futures = [
                loader_executor.submit(_load_image_safe, rec, dataset_loader.images_dir) 
                for rec in chunk_records
            ]
            
            batch_imgs = []
            batch_keys = []
            
            for f in futures:
                res = f.result()
                if res:
                    path, img, rec = res
                    batch_imgs.append(img)
                    batch_keys.append(_baseline_key(rec, path))
            
            #batch inference
            if batch_imgs:
                preds = inference_backend.predict_batch(batch_imgs)
                for k, pred in zip(batch_keys, preds):
                    baseline_ground_truth[k] = [
                        {"bbox": b, "category_id": l} for b, l in zip(pred.boxes, pred.labels)
                    ]
            
            print(f"  [Baseline] Processed {min(i + batch_size, total_images)}/{total_images}...", end="\r")
            
    print("")
    return baseline_ground_truth # type: ignore

class BenchmarkPipeline:
    """
    Orchestrates dataset iteration, on the fly modification, inference, evaluation.
    """
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        modifier: ModifierStrategy,
        inference_backend: InferenceBackend,
        storage_backend: StorageBackend,
        gpu_batch_size: int = 8,
        passes: int = 3,
        step_size_percent: float = 10.0,
        baseline_data: Optional[Dict[str, List[Dict]]] = None
    ):
        self.dataset_loader = dataset_loader
        self.modifier = modifier
        self.inference_backend = inference_backend
        self.storage_backend = storage_backend
        self.gpu_batch_size = gpu_batch_size
        self.passes = passes
        self.step_size_percent = step_size_percent
        self.baseline_ground_truth: Dict[str, List[Dict]] = baseline_data if baseline_data is not None else {}

        #inmemory storage for final mAP calculation
        self.preds_by_pass: Dict[int, List[InferenceResult]] = {p: [] for p in range(self.passes + 1)}
        self.gts_by_pass: Dict[int, List[List[Dict]]] = {p: [] for p in range(self.passes + 1)}
        
        self.metrics_accum: Dict[int, Dict[str, float]] = {p: {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0, "entropy": 0.0, "overexposure": 0.0} for p in range(self.passes + 1)}
        self.total_conf: Dict[int, float] = {p: 0.0 for p in range(self.passes + 1)}
        self.total_dets: Dict[int, int] = {p: 0 for p in range(self.passes + 1)}

        self.flush_future: Optional[Future] = None

    def _baseline_key(self, rec: ImageRecord, path: str) -> str:
        return _baseline_key(rec, path)

    def build_baseline_ground_truth(self) -> None:
        if not self.baseline_ground_truth:
             self.baseline_ground_truth = compute_baseline(self.dataset_loader, self.inference_backend, self.gpu_batch_size)

    def _get_gt_list(self, rec: ImageRecord, path: str) -> List[Dict]:
        key = self._baseline_key(rec, path)
        return self.baseline_ground_truth.get(key, [])

    def run(self) -> Dict[int, Dict]:
        """
        Runs the pipeline for the selected pass range.
        Now captures mAP, Avg Confidence, and Detection Count.
        """
        results: Dict[int, Dict] = {}

        sample_count = len(self.dataset_loader.images)

        if sample_count == 0:
            print("[pipeline] no readable images found")
            return results

        self.build_baseline_ground_truth()

        pass_range = range(0, self.passes + 1)

        # jsonl checkpoint file
        checkpoint_file = f"checkpoint_{self.modifier.name}.jsonl"
        
        print(f"[pipeline] Starting Pipelined Processing for {sample_count} images...")
        t_start = time.time()

        # pipelined execution
        total_cores = os.cpu_count() or 4
        gen_workers = max(1, int(total_cores * 0.75))
        metric_workers = max(1, total_cores - gen_workers - 1)
        
        print(f"[pipeline] Resource Allocation: Gen={gen_workers}, Metrics={metric_workers} (Total Cores: {total_cores})")
        
        gen_executor = ProcessPoolExecutor(max_workers=gen_workers)
        metric_executor = ThreadPoolExecutor(max_workers=metric_workers)
        flush_executor = ThreadPoolExecutor(max_workers=1)

        window_size = gen_workers * 2
        pending_futures = deque()
        data_iterator = enumerate(self.dataset_loader.iter_samples())

        def submit_next_task():
            try:
                idx, (path, img, rec) = next(data_iterator)

                future = gen_executor.submit(generate_batch_task, img, rec, pass_range, self.modifier) # type: ignore

                pending_futures.append((future, idx, path, rec))
                return True
            except StopIteration:
                return False

        print(f"[pipeline] filling prefetch queue (size={window_size})...")
        # fill queue
        for _ in range(window_size):
            if not submit_next_task():
                break

        # gpu batching
        GPU_BATCH_SIZE = self.gpu_batch_size
        batch_buffer = []
        
        #async flush state
        self.flush_future = None

        def flush_buffer_task(buffer_snapshot):
            """Executed in background thread"""
            if not buffer_snapshot:
                return
            
            #unpack buffer
            b_imgs = [b[0] for b in buffer_snapshot]
            b_passes = [b[1] for b in buffer_snapshot]
            b_metrics = [b[2] for b in buffer_snapshot]
            b_recs = [b[3] for b in buffer_snapshot]
            b_gts = [b[4] for b in buffer_snapshot]

            #batch inference
            b_preds = self.inference_backend.predict_batch(b_imgs)
            
            with open(checkpoint_file, 'a') as f_out:
                #process results
                for p, pred, img_metrics, rec, gt_list in zip(b_passes, b_preds, b_metrics, b_recs, b_gts):
                    self.preds_by_pass[p].append(pred)
                    self.gts_by_pass[p].append(gt_list)

                    n_dets = len(pred.scores)
                    self.total_dets[p] += n_dets
                    if n_dets > 0:
                        self.total_conf[p] += float(sum(pred.scores, 0.0))
                        
                    checkpoint_data = {
                        "image_id": rec.id,
                        "file_name": rec.file_name,
                        "pass": p,
                        "metrics": img_metrics,
                        "detections": n_dets
                    }
                    f_out.write(json.dumps(checkpoint_data) + "\n")

        def trigger_async_flush(current_buffer):
            if self.flush_future and not self.flush_future.done():
                self.flush_future.result()
            
            buffer_snapshot = list(current_buffer)
            current_buffer.clear()
            
            self.flush_future = flush_executor.submit(flush_buffer_task, buffer_snapshot)

        # process queue
        while pending_futures:
            future, img_index, path, rec = pending_futures.popleft()
            
            submit_next_task()
            
            try:
                batch_data = future.result()
            except Exception as e:
                print(f"[pipeline] Error in generation task for {path}: {e}")
                continue

            gt_list = self._get_gt_list(rec, path)
            
            # parallel metrics calculation
            results_map: Dict[int, Dict[str, float]] = {} 
            
            #use persistent executor
            future_to_pass = {
                metric_executor.submit(measure_all, img): p 
                for p, img in batch_data
            }
        
            #collect results as they complete
            for future in as_completed(future_to_pass):
                p = future_to_pass[future]
                try:
                    results_map[p] = future.result() # type: ignore
                except Exception as e:
                    print(f"[pipeline] Error calculating metrics for pass {p}: {e}")
                    results_map[p] = measure_all(batch_data[p][1])

            #accumulate into buffer
            for p, mod_img in batch_data:
                img_metrics = results_map.get(p, {})
                for k, v in img_metrics.items():
                    self.metrics_accum[p][k] += v

                self.storage_backend.save(mod_img, self.modifier.name, p, rec.file_name)
                
                #add to gpu buffer
                batch_buffer.append((mod_img, p, img_metrics, rec, gt_list))

            #flush if full
            if len(batch_buffer) >= GPU_BATCH_SIZE:
                trigger_async_flush(batch_buffer)

            if (img_index + 1) % 10 == 0:
                print(f"  Processed {img_index + 1}/{sample_count} images...", end="\r")
        
        #final flush
        trigger_async_flush(batch_buffer)
        if self.flush_future:
            self.flush_future.result()
        
        gen_executor.shutdown()
        metric_executor.shutdown()
        flush_executor.shutdown()

        t_end = time.time()
        print(f"\n[pipeline] All images processed in {t_end - t_start:.1f}s. Computing final mAP...")

        #calculate final aggregated metrics per pass
        for p in pass_range:
            mAP = evaluate_map_at_iou(self.preds_by_pass[p], self.gts_by_pass[p], iou_thresh=0.5)
            
            #averages
            avg_metrics = {k: v / sample_count for k, v in self.metrics_accum[p].items()}
            
            #global average confidence
            avg_conf = (self.total_conf[p] / self.total_dets[p]) if self.total_dets[p] > 0 else 0.0
            
            #average detections per image
            avg_count = self.total_dets[p] / sample_count

            print(f"[pipeline] pass {p}: mAP={mAP:.3f} | Confidence={avg_conf:.3f} | Detection Count={avg_count:.1f}")

            results[p] = {
                "score": mAP,
                "avg_confidence": avg_conf,
                "avg_detection_count": avg_count,
                "metrics": avg_metrics
            }

        return results

    @staticmethod
    def plot_results(results: Dict[int, float], step_percent: float):
        import matplotlib.pyplot as plt
        import math
        xs = []
        ys = []
        for p in sorted(results.keys()):
            v = results[p]
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            xs.append(p * step_percent)
            ys.append(v)

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Total Degradation Percent")
        plt.ylabel("mAP@0.5")
        plt.title("Performance vs Degradation")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", "performance_curve.png")

        plt.savefig(save_path)
        print(f"[plot] Saved results plot to {save_path}")
        plt.close()