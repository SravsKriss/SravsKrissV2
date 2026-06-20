import subprocess
import os

class FFMPEGExport:
    @staticmethod
    def export_video(input_video: str, frames_dir: str, output_path: str, fps: float, audio_source: str = None):
        """Combines frames and optional audio into a final video using FFmpeg."""
        if not os.path.exists(frames_dir):
            return False
            
        # Base command for inputs
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%06d.jpg')
        ]
        
        # If audio source is provided, add it as second input
        if audio_source and os.path.exists(audio_source):
            cmd.extend(['-i', audio_source])
            has_audio = True
        else:
            has_audio = False
            
        # Encoding parameters
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p'
        ])
        
        if has_audio:
            cmd.extend([
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest'
            ])
        else:
            # No audio mapping needed
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
