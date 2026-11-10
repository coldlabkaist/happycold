"""PNG mask I/O supporting Unicode paths on Windows."""
from pathlib import Path

import cv2
import numpy as np


def read_mask_png(path: str | Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None
    except (OSError, ValueError, cv2.error):
        return None


def write_mask_png(path: str | Path, image: np.ndarray) -> bool:
    try:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return False
        encoded.tofile(path)
        return True
    except (OSError, ValueError, cv2.error):
        return False
