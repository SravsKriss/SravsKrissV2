# DanceTrack AI 💃

DanceTrack AI is a professional-grade video tracking and virtual camera application. It automatically follows a dancer or a selected target in a video, creating a smooth, cinematic virtual camera movement that keeps the subject centered.

## 🌟 Features

- **AI Auto Tracking**: Uses MediaPipe Pose to automatically detect and follow a dancer's center of mass.
- **Manual Camera Tracking**: High-precision CSRT tracker for following specific objects, heads, hands, or props.
- **Manual Keyframe Camera Mode**: Full creative control to manually place camera focus points and interpolate paths using Cubic Splines.
- **Cinematic Smoothing**: Professional filters (Savitzky-Golay, Moving Average) to remove camera shake.
- **Virtual Camera Engine**: 
  - Precision cropping for 9:16 (Shorts/TikTok), 16:9, and 1:1 ratios.
  - Adjustable Zoom and Follow Strength.
  - Automatic Audio preservation.
- **Export**: High-quality H.264 export using FFmpeg.

## 🛠️ Tech Stack

- **Logic**: Python 3.12, OpenCV, MediaPipe Pose, NumPy, SciPy
- **Rendering**: FFmpeg
- **UI**: Streamlit

## 🚀 Installation

### 1. Requirements
Ensure you have Python 3.12+ installed.

### 2. FFmpeg Setup
This application requires FFmpeg to be installed on your system and added to your PATH.
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`.
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 3. Setup Virtual Environment (Recommended)
To avoid dependency conflicts (especially with MediaPipe), it is highly recommended to use a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install Dependencies
If not using a venv:
```bash
pip install -r requirements.txt
```

## 🎥 Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. **Upload** your dance video (MP4, MOV, or AVI).
3. **Choose Tracking Mode**:
   - **AI Auto**: Best for full-body dance videos.
   - **Manual**: Best for tracking a specific face or object.
   - **Keyframe**: Best for complex scenes requiring manual focus points.
4. **Adjust Settings**: Fine-tune zoom, aspect ratio, and smoothness in the sidebar.
5. **Process**: Click the "Process" button and wait for the AI to work its magic.
6. **Download**: Preview the side-by-side result and download your final video.

## 📁 Project Structure

```text
DanceAI/
├── app.py              # Streamlit Interface
├── tracker/
│   ├── pose_detector.py # AI Pose Detection
│   ├── manual_tracker.py# CSRT Tracking
│   ├── camera_path.py   # Path Interpolation
│   └── video_processor.py# Virtual Camera Engine
├── utils/
│   ├── ffmpeg_export.py # FFmpeg Integration
│   ├── smoothing.py     # Path Smoothing Filters
│   └── file_handler.py  # File Utilities
├── outputs/            # Processed Videos
└── requirements.txt    # Project Dependencies
```

## 📝 Success Criteria
The application aims to provide an experience comparable to or better than CapCut's AI Camera Tracking, specifically optimized for dancers.

---
Created with ❤️ by Antigravity
