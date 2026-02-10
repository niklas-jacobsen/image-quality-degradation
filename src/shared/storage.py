import os
import cv2 as cv
import numpy as np
from typing import Protocol, Optional

class StorageBackend(Protocol):
    def save(self, image: np.ndarray, modifier: str, level: int, filename: str) -> Optional[str]:
        ...
    def should_persist(self) -> bool:
        ...

class DiskBackend:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def save(self, image: np.ndarray, modifier: str, level: int, filename: str) -> str:
        #structure: root/modifier/level_N/filename
        dir_path = os.path.join(self.root_dir, modifier, f"level_{level}")
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
        cv.imwrite(full_path, image)
        return full_path

    def should_persist(self) -> bool:
        return True

class EphemeralBackend:
    """RAM Mode: does not save images to a list. Used for on-the-fly metrics."""
    def save(self, image: np.ndarray, modifier: str, level: int, filename: str) -> Optional[str]:
        return None

    def should_persist(self) -> bool:
        return False
