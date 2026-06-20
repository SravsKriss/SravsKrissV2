import cv2
import numpy as np

class ColorEnhanceModule:
    @staticmethod
    def apply(frame: np.ndarray, contrast: float = 1.1, saturation: float = 1.2, brightness: int = 5) -> np.ndarray:
        """Applies contrast, saturation and brightness adjustments."""
        # 1. Brightness & Contrast
        # new_image = alpha*image + beta
        enhanced = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # 2. Saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv2.split(hsv)
        s = s * saturation
        s = np.clip(s, 0, 255)
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        
        return enhanced
