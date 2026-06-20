import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class ManualTracker:
    def __init__(self):
        self.tracker = None
        self.initialized = False
        self.bbox = None

    def init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initializes the CSRT tracker with a bounding box (x, y, w, h)."""
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = self.tracker.init(frame, bbox)
        self.bbox = bbox
        return self.initialized

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """Updates the tracker and returns success and center point."""
        if not self.initialized:
            return False, None
            
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w / 2
            center_y = y + h / 2
            
            return True, {
                "x": center_x,
                "y": center_y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox": (x, y, w, h)
            }
        else:
            return False, None

    def reset(self):
        self.tracker = None
        self.initialized = False
        self.bbox = None
