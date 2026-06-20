import cv2
import numpy as np
from enhancement.denoise import DenoiseModule
from enhancement.sharpen import SharpenModule
from enhancement.color_enhance import ColorEnhanceModule
from enhancement.face_enhance import FaceEnhanceModule
from enhancement.upscale import UpscaleModule

class EnhancementManager:
    def __init__(self, settings: dict):
        self.settings = settings
        self.upscaler = UpscaleModule() if settings.get('enable_ai_upscaling') else None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies enabled enhancement modules to a frame."""
        processed = frame.copy()
        
        # 1. Denoise
        if self.settings.get('enable_denoising'):
            processed = DenoiseModule.apply(processed)
            
        # 2. Color Enhancement
        if self.settings.get('enable_color_enhancement'):
            processed = ColorEnhanceModule.apply(processed)
            
        # 3. Sharpening
        if self.settings.get('enable_sharpening'):
            processed = SharpenModule.apply(processed)
            
        # 4. Face Enhancement
        if self.settings.get('enable_face_enhancement'):
            processed = FaceEnhanceModule.apply(processed)
            
        # 5. Upscaling
        if self.settings.get('enable_ai_upscaling') and self.upscaler:
            processed = self.upscaler.upscale(processed)
            
        # Face enhancement usually requires a specialized model (GFPGAN)
        # For now, it's bypassed unless the user integrates the GFPGAN package.
        
        return processed
