import os
import subprocess

class MotionSmoothingModule:
    @staticmethod
    def apply_ffmpeg_interpolation(video_path, output_path, target_fps=60):
        """Uses FFmpeg's minterpolate filter to increase framerate (Motion Smoothing)."""
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-filter:v', f'minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir',
            '-c:a', 'copy',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception:
            return False
