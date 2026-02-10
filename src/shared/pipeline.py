from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
import cv2 as cv
import os
import time
from dataclasses import dataclass

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

class DatasetLoader:
    """
    Loads images from disk.
    """

    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self.images: Dict[int, ImageRecord] = {}
        
        for idx, fname in enumerate(sorted(os.listdir(images_dir))):
            path = os.path.join(images_dir, fname)
            if not os.path.isfile(path):
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
        passes: int = 3,
        step_size_percent: float = 10.0,
    ):
        self.dataset_loader = dataset_loader
        self.modifier = modifier
        self.inference_backend = inference_backend
        self.storage_backend = storage_backend
        self.passes = passes
        self.step_size_percent = step_size_percent
        self.baseline_ground_truth: Dict[str, List[Dict]] = {}

    def _baseline_key(self, rec: ImageRecord, path: str) -> str:
        return rec.file_name if rec.file_name else path


    def build_baseline_ground_truth(self) -> None:
        self.baseline_ground_truth = {}
        for path, img, rec in self.dataset_loader.iter_samples():
            key = self._baseline_key(rec, path)
            pred = self.inference_backend.predict(img)
            self.baseline_ground_truth[key] = [
                {"bbox": b, "category_id": l} for b, l in zip(pred.boxes, pred.labels)
            ]


    def _get_gt_list(self, rec: ImageRecord, path: str) -> List[Dict]:
        key = self._baseline_key(rec, path)
        return self.baseline_ground_truth.get(key, [])

    def run(self) -> Dict[int, Dict]:
        """
        Runs the pipeline for the selected pass range.
        Now captures mAP, Avg Confidence, and Detection Count.
        """
        results: Dict[int, Dict] = {}

        #materialize samples once so indexing is stable across passes
        samples = list(self.dataset_loader.iter_samples())
        if not samples:
            print("[pipeline] no readable images found")
            return results

        print("[pipeline] Generating baseline ground truth from Pass 0 (Originals)...")
        self.build_baseline_ground_truth()

        pass_range = range(0, self.passes + 1)

        for p in pass_range:
            print(f"[pipeline] running pass {p}/{self.passes}")

            preds_all: List[InferenceResult] = []
            gts_all: List[List[Dict]] = []
            
            metrics_accum = {"brightness": 0.0, "contrast": 0.0, "sharpness": 0.0, "entropy": 0.0, "overexposure": 0.0}
            total_conf = 0.0
            total_dets = 0
            sample_count = len(samples)

            t0 = time.time()

            for img_index, (path, img, rec) in enumerate(samples):
                #apply no modifier in baseline generation pass only
                mod_img = img.copy() if p == 0 else self.modifier.apply(img, p)

                #measure image quality metrics
                img_metrics = measure_all(mod_img)
                for k, v in img_metrics.items():
                    metrics_accum[k] += v

                self.storage_backend.save(mod_img, self.modifier.name, p, rec.file_name)
                
                #predict
                pred = self.inference_backend.predict(mod_img)
                preds_all.append(pred)

                n_dets = len(pred.scores)
                total_dets += n_dets
                if n_dets > 0:
                    total_conf += float(sum(pred.scores, 0.0))

                gt_list = self._get_gt_list(rec, path)
                gts_all.append(gt_list)

            t1 = time.time()
            print(f"[pipeline] inference done for pass {p}, time {t1 - t0:.1f}s, computing metrics")

            mAP = evaluate_map_at_iou(preds_all, gts_all, iou_thresh=0.5)
            
            #calculate averages
            avg_metrics = {k: v / sample_count for k, v in metrics_accum.items()}
            
            #global average confidence (sum of all confidences/total number of detections)
            avg_conf = (total_conf / total_dets) if total_dets > 0 else 0.0
            
            #average detections per image (total dets/number of images)
            avg_count = total_dets / sample_count

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
