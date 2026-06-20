import cv2
import numpy as np
from typing import Dict, Any

class QualityAnalyzer:
    @staticmethod
    def estimate_quality(crop_width: int, crop_height: int, target_width: int, target_height: int) -> Dict[str, Any]:
        """Estimates the output quality based on the crop-to-target ratio."""
        zoom_level = target_width / crop_width if crop_width > 0 else 1.0
        
        if zoom_level <= 1.2:
            rating = "Excellent"
            color = "green"
            suggestion = "No upscaling required."
        elif zoom_level <= 2.0:
            rating = "Good"
            color = "blue"
            suggestion = "Sharpening recommended."
        elif zoom_level <= 3.0:
            rating = "Fair"
            color = "orange"
            suggestion = "AI Upscaling suggested for better detail."
        else:
            rating = "Low"
            color = "red"
            suggestion = "AI Upscaling highly recommended to fix pixelation."
            
        return {
            "rating": rating,
            "color": color,
            "zoom_level": f"{zoom_level:.2f}x",
            "suggestion": suggestion
        }
