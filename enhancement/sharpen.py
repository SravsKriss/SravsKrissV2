import cv2
import numpy as np

class SharpenModule:
    @staticmethod
    def apply(frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Applies an unsharp mask filter for sharpening."""
        # strength: [1.0 - 2.0]
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, strength, blurred, 1.0 - strength, 0)
        return sharpened
