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
        self.original_settings = settings.copy()

    def analyze_and_refine_settings(self, input_res: tuple, crop_res: tuple, target_res: tuple, fps: float):
        """Smartly determines which modules are actually necessary."""
        # 1. Skip upscaling if crop resolution is high enough
        # If crop_width >= 0.8 * target_width, upscaling is likely a waste of time/resources
        crop_w, crop_h = crop_res
        target_w, target_h = target_res
        
        if crop_w >= target_w * 0.8:
            if self.settings.get('enable_ai_upscaling'):
                self.settings['enable_ai_upscaling'] = False
                # Optionally keep sharpening if upscaling was disabled
                self.settings['enable_sharpening'] = self.original_settings.get('enable_sharpening', True)

        # 2. Skip RIFE (Motion Smoothing) if input FPS is already high
        if fps >= 55: # Close to 60
            self.settings['enable_motion_smoothing'] = False

        # 3. Skip Face Enhancement if crop resolution is too low for it to be effective
        # Minimum reasonable width for face enhancement to work well
        if crop_w < 300:
            self.settings['enable_face_enhancement'] = False

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
