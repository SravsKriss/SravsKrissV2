import cv2
import numpy as np

class DenoiseModule:
    @staticmethod
    def apply(frame: np.ndarray, strength: int = 7) -> np.ndarray:
        """Applies fast non-local means denoising."""
        # strength: [1-20]
        # Using fastNlMeansDenoisingColored for speed
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
