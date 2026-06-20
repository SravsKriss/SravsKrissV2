import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Callable
from tracker.pose_detector import PoseDetector
from tracker.manual_tracker import ManualTracker
from tracker.camera_path import CameraPath
from utils.smoothing import SmoothingUtils
from utils.ffmpeg_export import FFMPEGExport
from enhancement.enhancement_manager import EnhancementManager
from enhancement.motion_smoothing import MotionSmoothingModule

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str = "outputs"):
        self.video_path = video_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.temp_frames_dir = os.path.join(output_dir, "temp_frames")
        if not os.path.exists(self.temp_frames_dir):
            os.makedirs(self.temp_frames_dir)

    def process_tracking(self, mode: str, 
                         progress_callback: Callable = None, 
                         bbox: Tuple = None) -> CameraPath:
        """Runs the tracking process based on the selected mode."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        path_manager = CameraPath(self.total_frames)
        
        if mode == "AI Auto Tracking":
            detector = PoseDetector()
            for i in range(self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                _, center = detector.find_pose(frame, draw=False)
                if center:
                    path_manager.add_raw_point(i, center)
                
                if progress_callback:
                    progress_callback(i / self.total_frames, f"AI Tracking frame {i}/{self.total_frames}")
            detector.close()
            
        elif mode == "Manual Camera Tracking" and bbox:
            tracker = ManualTracker()
            detector = PoseDetector() # AI assistance
            
            ret, first_frame = self.cap.read()
            if not ret:
                return path_manager
                
            tracker.init_tracker(first_frame, bbox)
            # Add first point
            x, y, w, h = bbox
            path_manager.add_raw_point(0, {"x": x + w/2, "y": y + h/2, "bbox_w": w, "bbox_h": h})
            
            for i in range(1, self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 1. Update Manual Tracker
                success_manual, manual_data = tracker.update(frame)
                
                # 2. Update with AI Assistance
                # Try to find a pose in the whole frame
                _, ai_data = detector.find_pose(frame, draw=False)
                
                final_center = None
                if ai_data and success_manual:
                    # Check if AI center is roughly within the manual box (leeway 1.5x box size)
                    mx, my = manual_data['x'], manual_data['y']
                    ax, ay = ai_data['x'], ai_data['y']
                    dist = ((mx - ax)**2 + (my - ay)**2)**0.5
                    
                    # If AI center is close to manual box, trust AI (it's smoother/more centered)
                    if dist < max(manual_data['bbox_w'], manual_data['bbox_h']):
                        final_center = ai_data
                    else:
                        final_center = manual_data
                elif ai_data:
                    # Manual failed, but AI found someone? 
                    # If they were close to the last known position, re-identify
                    final_center = ai_data
                else:
                    final_center = manual_data
                
                if final_center:
                    path_manager.add_raw_point(i, final_center)
                elif not success_manual:
                    # Both failed
                    break
                
                if progress_callback:
                    progress_callback(i / self.total_frames, f"AI-Assisted Tracking frame {i}/{self.total_frames}")
            detector.close()
                    
        return path_manager

    def generate_virtual_camera(self, path_manager: CameraPath, 
                                 aspect_ratio: str = "9:16",
                                 zoom_factor: float = 1.2,
                                 smoothness: float = 0.9,
                                 smoothing_method: str = "savgol",
                                 interpolation_method: str = "cubic",
                                 is_keyframe_mode: bool = False,
                                 progress_callback: Callable = None) -> str:
        """Generates the final video with virtual camera movement."""
        
        # 1. Get path
        if is_keyframe_mode:
            raw_path = path_manager.interpolate_keyframes(method=interpolation_method)
        else:
            raw_path = path_manager.get_raw_path()
            
        if not raw_path:
            return ""
            
        # 2. Smooth path
        # Map smoothness 0-1 to window size (roughly)
        window_size = int(max(3, (1 - smoothness) * 100))
        if window_size % 2 == 0: window_size += 1
        
        smoothed_path = SmoothingUtils.smooth_path(raw_path, window_size=window_size, method=smoothing_method)
        
        # 2b. Handle NaN or empty smoothed path
        if not smoothed_path:
            # Fallback to centering on the frame
            smoothed_path = [(self.width/2, self.height/2)] * self.total_frames
        else:
            # Replace NaNs with frame center
            new_path = []
            last_valid = (self.width/2, self.height/2)
            for x, y in smoothed_path:
                if np.isnan(x) or np.isnan(y):
                    new_path.append(last_valid)
                else:
                    last_valid = (x, y)
                    new_path.append(last_valid)
            smoothed_path = new_path
        
        # 3. Calculate target resolution based on aspect ratio
        target_w, target_h = self._get_target_dims(aspect_ratio)
        
        # 4. Render frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        output_filename = os.path.join(self.output_dir, "tracked_output.mp4")
        
        for i in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            center_x, center_y = smoothed_path[i] if i < len(smoothed_path) else smoothed_path[-1]
            
            # Smart Crop Logic
            # Calculate current crop box based on zoom_factor
            # Base zoom: how much of the original height/width we use
            crop_h = self.height / zoom_factor
            crop_w = crop_h * (target_w / target_h)
            
            if crop_w > self.width:
                crop_w = self.width
                crop_h = crop_w * (target_h / target_w)
                
            # Center the crop box on (center_x, center_y)
            x1 = int(center_x - crop_w / 2)
            y1 = int(center_y - crop_h / 2)
            x2 = int(x1 + crop_w)
            y2 = int(y1 + crop_h)
            
            # Clamp to frame boundaries
            if x1 < 0:
                dx = -x1
                x1 += dx; x2 += dx
            if y1 < 0:
                dy = -y1
                y1 += dy; y2 += dy
            if x2 > self.width:
                dx = x2 - self.width
                x1 -= dx; x2 -= dx
            if y2 > self.height:
                dy = y2 - self.height
                y1 -= dy; y2 -= dy
                
            # Crop and resize
            cropped = frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)]
            if cropped.size == 0:
                cropped = frame # Fallback
                
            processed = cv2.resize(cropped, (target_w, target_h))
            
            frame_name = os.path.join(self.temp_frames_dir, f"{i:06d}.jpg")
            cv2.imwrite(frame_name, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if progress_callback:
                progress_callback((i + 1) / self.total_frames, f"Rendering frame {i + 1}/{self.total_frames}")

        # 5. Extract audio and Export
        if progress_callback:
            progress_callback(0.99, "Extracting audio...")
            
        audio_temp = os.path.join(self.output_dir, "temp_audio.aac")
        FFMPEGExport.extract_audio(self.video_path, audio_temp)
        
        if progress_callback:
            progress_callback(0.99, "FFmpeg Exporting final video... (this may take a minute)")
        
        success = FFMPEGExport.export_video(
            input_video=self.video_path,
            frames_dir=self.temp_frames_dir,
            output_path=output_filename,
            fps=self.fps,
            audio_source=audio_temp if os.path.exists(audio_temp) else None
        )
        
        if progress_callback:
            if success:
                progress_callback(1.0, "Export Complete!")
            else:
                progress_callback(1.0, "Export Failed in FFmpeg phase.")
        
        # Cleanup
        if os.path.exists(audio_temp):
            os.remove(audio_temp)
            
        self.cap.release()
        return output_filename if success else ""

    def generate_enhanced_video(self, path_manager: CameraPath, 
                                 aspect_ratio: str = "9:16",
                                 zoom_factor: float = 1.2,
                                 smoothness: float = 0.9,
                                 smoothing_method: str = "savgol",
                                 interpolation_method: str = "cubic",
                                 is_keyframe_mode: bool = False,
                                 enhance_settings: dict = None,
                                 progress_callback: Callable = None) -> str:
        """Generates the final video with virtual camera movement and enhancement modules."""
        
        # 1. Get path and smoothing (same as standard)
        if is_keyframe_mode:
            raw_path = path_manager.interpolate_keyframes(method=interpolation_method)
        else:
            raw_path = path_manager.get_raw_path()
            
        if not raw_path:
            return ""
            
        window_size = int(max(3, (1 - smoothness) * 100))
        if window_size % 2 == 0: window_size += 1
        smoothed_path = SmoothingUtils.smooth_path(raw_path, window_size=window_size, method=smoothing_method)
        
        if not smoothed_path:
            smoothed_path = [(self.width/2, self.height/2)] * self.total_frames
        else:
            new_path = []
            last_valid = (self.width/2, self.height/2)
            for x, y in smoothed_path:
                if np.isnan(x) or np.isnan(y):
                    new_path.append(last_valid)
                else:
                    last_valid = (x, y)
                    new_path.append(last_valid)
            smoothed_path = new_path
        
        target_w, target_h = self._get_target_dims(aspect_ratio)
        
        # 2. Initialize Enhancement Manager
        manager = EnhancementManager(enhance_settings or {})
        
        # 3. Render and Enhance frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        output_filename = os.path.join(self.output_dir, "enhanced_output.mp4")
        
        for i in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            center_x, center_y = smoothed_path[i] if i < len(smoothed_path) else smoothed_path[-1]
            
            crop_h = self.height / zoom_factor
            crop_w = crop_h * (target_w / target_h)
            
            if crop_w > self.width:
                crop_w = self.width
                crop_h = crop_w * (target_h / target_w)
                
            x1 = int(center_x - crop_w / 2)
            y1 = int(center_y - crop_h / 2)
            x2 = int(x1 + crop_w)
            y2 = int(y1 + crop_h)
            
            # Clamp
            if x1 < 0: dx = -x1; x1 += dx; x2 += dx
            if y1 < 0: dy = -y1; y1 += dy; y2 += dy
            if x2 > self.width: dx = x2 - self.width; x1 -= dx; x2 -= dx
            if y2 > self.height: dy = y2 - self.height; y1 -= dy; y2 -= dy
                
            cropped = frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)]
            if cropped.size == 0: cropped = frame
            
            resized = cv2.resize(cropped, (target_w, target_h))
            
            # --- Enhancement Step ---
            processed = manager.process_frame(resized)
            
            frame_name = os.path.join(self.temp_frames_dir, f"{i:06d}.jpg")
            cv2.imwrite(frame_name, processed, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            if progress_callback:
                progress_callback((i + 1) / self.total_frames, f"Enhancing & Rendering frame {i + 1}/{self.total_frames}")

        # 4. Extract audio and Export
        audio_temp = os.path.join(self.output_dir, "temp_audio.aac")
        FFMPEGExport.extract_audio(self.video_path, audio_temp)
        
        codec = enhance_settings.get('export_codec', 'libx264')
        if codec == 'H265': codec = 'libx265'
        else: codec = 'libx264'
        
        success = FFMPEGExport.export_video(
            input_video=self.video_path,
            frames_dir=self.temp_frames_dir,
            output_path=output_filename,
            fps=self.fps,
            audio_source=audio_temp if os.path.exists(audio_temp) else None
            # Future: add codec support to export_video if needed
        )
        
        # 5. Optional Motion Smoothing (60 FPS)
        if success and enhance_settings.get('enable_motion_smoothing'):
            if progress_callback:
                progress_callback(0.99, "Applying Motion Smoothing (RIFE placeholder)...")
            
            smooth_output = os.path.join(self.output_dir, "smoothed_enhanced_output.mp4")
            if MotionSmoothingModule.apply_ffmpeg_interpolation(output_filename, smooth_output):
                output_filename = smooth_output
        
        if os.path.exists(audio_temp):
            os.remove(audio_temp)
            
        self.cap.release()
        return output_filename if success else ""

    def _get_target_dims(self, ratio: str) -> Tuple[int, int]:
        if ratio == "9:16": return (720, 1280)
        if ratio == "16:9": return (1280, 720)
        if ratio == "1:1": return (1080, 1080)
        return (self.width, self.height)
