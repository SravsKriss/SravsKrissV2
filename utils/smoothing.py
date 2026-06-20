import numpy as np
from scipy.signal import savgol_filter
from typing import List, Tuple

class SmoothingUtils:
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Applies a moving average filter to the data."""
        if window_size <= 1:
            return data
        
        # Ensure window size is odd for symmetry
        if window_size % 2 == 0:
            window_size += 1
            
        pad_width = window_size // 2
        extended_data = np.pad(data, (pad_width, pad_width), mode='edge')
        smoothed = np.convolve(extended_data, np.ones(window_size)/window_size, mode='valid')
        return smoothed

    @staticmethod
    def savitzky_golay(data: np.ndarray, window_size: int, poly_order: int = 2) -> np.ndarray:
        """Applies a Savitzky-Golay filter to the data."""
        if len(data) < window_size:
            window_size = len(data) if len(data) % 2 != 0 else len(data) - 1
            
        if window_size < 3:
            return data
            
        # Ensure window_size is odd and greater than poly_order
        if window_size % 2 == 0:
            window_size += 1
        
        if window_size <= poly_order:
            poly_order = window_size - 1
            
        try:
            return savgol_filter(data, window_size, poly_order)
        except Exception:
            return data

    @staticmethod
    def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Applies exponential smoothing to the data."""
        if not 0 < alpha <= 1:
            return data
            
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for t in range(1, len(data)):
            smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
        return smoothed

    @classmethod
    def smooth_path(cls, path: List[Tuple[float, float]], window_size: int = 15, method: str = "savgol") -> List[Tuple[float, float]]:
        """Smooths a 2D path using the specified method."""
        if not path:
            return []
            
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        if method == "savgol":
            sx = cls.savitzky_golay(x, window_size)
            sy = cls.savitzky_golay(y, window_size)
        elif method == "moving_average":
            sx = cls.moving_average(x, window_size)
            sy = cls.moving_average(y, window_size)
        elif method == "exponential":
            sx = cls.exponential_smoothing(x, 0.2)
            sy = cls.exponential_smoothing(y, 0.2)
        else:
            sx, sy = x, y
            
        return list(zip(sx, sy))
