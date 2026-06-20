import os
import cv2
from typing import Dict, Any, Optional
import shutil

class FileHandler:
    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extracts metadata from a video file."""
        if not os.path.exists(video_path):
            return {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "filename": os.path.basename(video_path),
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": f"{duration:.2f}s"
        }

    @staticmethod
    def save_uploaded_file(uploaded_file, directory: str = "temp") -> str:
        """Saves an uploaded file to a temporary directory."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    @staticmethod
    def clear_temp(directory: str = "temp"):
        """Clears the temporary directory."""
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
