# DanceTrack AI Architecture

The system is designed with a clear separation between the UI (Streamlit), the Tracking Logic, and the Rendering Engine (FFmpeg).

## System Flow

```mermaid
graph TD
    A[User Uploads Video] --> B{Streamlit UI}
    B --> C[File Handler]
    C --> D[Video Metadata]
    
    B --> E{Tracking Mode Selection}
    
    E -- AI Auto --> F[Pose Detector - MediaPipe]
    E -- Manual --> G[CSRT Tracker - OpenCV]
    E -- Keyframe --> H[Keyframe Interpolator]
    
    F --> I[Raw Tracking Points]
    G --> I
    H --> J[Interpolated Path]
    
    I --> K[Smoothing Filters - Savitzky-Golay]
    J --> K
    
    K --> L[Virtual Camera Engine]
    L --> M[Crop & Resize Frames]
    M --> N[FFmpeg Export]
    N --> O[Final MP4 with Audio]
    O --> B
```

## Component Description

- **Pose Detector**: Uses MediaPipe to find 33 body landmarks. Calculates the center point based on shoulders and hips.
- **Manual Tracker**: Uses OpenCV's CSRT for robust object tracking.
- **Camera Path Manager**: Handles data storage and interpolates between sparse keyframes using Cubic Splines.
- **Smoothing Utils**: Implements signal processing filters to remove camera shake.
- **Video Processor**: Calculates crop coordinates frame-by-frame and prepares them for final encode.
- **FFmpeg Exporter**: Manages final assembly of processed frames and audio.
