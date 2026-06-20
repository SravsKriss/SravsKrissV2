import subprocess
import os

class FFMPEGExport:
    @staticmethod
    def export_video(input_video: str, frames_dir: str, output_path: str, fps: float, 
                     audio_source: str = None, preset: str = "medium", 
                     codec: str = "H264", crf: int = 23):
        """Combines frames and optional audio into a final video using FFmpeg with hardware acceleration."""
        if not os.path.exists(frames_dir):
            return False
            
        from utils.device_manager import device_manager
        gpu_info = device_manager.get_gpu_info()
        
        # Base command for inputs
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%06d.jpg')
        ]
        
        if audio_source and os.path.exists(audio_source):
            cmd.extend(['-i', audio_source])
            has_audio = True
        else:
            has_audio = False
            
        # Determine Encoder
        vcodec = "libx264"
        if codec == "H265":
            vcodec = "libx265"
            
        # Hardware Acceleration Check
        hw_accel = False
        if gpu_info['device'] == "cuda":
            if codec == "H264": vcodec = "h264_nvenc"; hw_accel = True
            elif codec == "H265": vcodec = "hevc_nvenc"; hw_accel = True
        elif gpu_info['gpu_type'] == "DirectML": # Could try AMF or QSV
            # Placeholder for AMF/QSV detection - for now fallback to CPU for safety unless user forces
            pass

        # Encoding parameters
        cmd.extend(['-c:v', vcodec])
        
        if hw_accel:
            # NVENC uses constant quality with different flags
            cmd.extend(['-rc:v', 'vbr', '-cq:v', str(crf), '-preset', preset if preset in ['slow', 'medium', 'fast'] else 'fast'])
        else:
            cmd.extend(['-preset', preset, '-crf', str(crf)])
            
        cmd.extend(['-pix_fmt', 'yuv420p'])
        
        if has_audio:
            cmd.extend(['-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest'])
        else:
            cmd.extend(['-map', '0:v:0'])
            
        cmd.append(output_path)
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode()
            with open("ffmpeg_error.log", "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Error: {error_msg}\n")
            print(f"FFmpeg error: {error_msg}")
            return False
        except Exception as e:
            with open("ffmpeg_error.log", "w") as f:
                f.write(f"Unexpected error: {str(e)}\n")
            return False

    @staticmethod
    def extract_audio(video_path: str, output_audio_path: str):
        """Extracts audio from a video file."""
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'copy',
            output_audio_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception:
            # If copy fails, try converting to aac
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'aac',
                output_audio_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except Exception:
                return False
