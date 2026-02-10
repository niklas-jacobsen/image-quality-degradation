from typing import List, Protocol
import numpy as np

#optional imports for inference
try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

class InferenceResult:
    """
    A simple container for detection outputs.
    boxes: list of [x1,y1,x2,y2]
    scores: list of float
    labels: list of int (category ids or label indices)
    """
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]

    def __init__(self, boxes: List[List[float]], scores: List[float], labels: List[int]):
        self.boxes = boxes
        self.scores = scores
        self.labels = labels

class InferenceBackend(Protocol):
    """
    Protocol for inference backends
    """
    def predict(self, img: np.ndarray) -> InferenceResult:
        ...

class YOLOInference:
    """
    Uses YOLO to run inference on a numpy image
    """
    def __init__(self, model_path: str = "yolov8s.pt", conf: float = 0.25):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics package not available.")
        self.model = YOLO(model_path)
        self.conf = conf

        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.label_to_name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            self.label_to_name = {i: str(n) for i, n in enumerate(names)}
        else:
            self.label_to_name = {}

    def predict(self, img: np.ndarray) -> InferenceResult:
        # model accepts numpy BGR images
        results = self.model(img, conf=self.conf, verbose=False)
        r = results[0]
        boxes = []
        scores = []
        labels = []
        for box in r.boxes:
            xyxy = box.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            boxes.append([float(x) for x in xyxy])
            scores.append(float(box.conf.tolist()[0]))
            labels.append(int(box.cls.tolist()[0]))
        return InferenceResult(boxes=boxes, scores=scores, labels=labels)
