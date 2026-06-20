import streamlit as st
import cv2
import os
import time
import numpy as np
import shutil
from PIL import Image
from streamlit_cropper import st_cropper
from utils.file_handler import FileHandler

# Check for FFmpeg
ffmpeg_available = shutil.which("ffmpeg") is not None
if not ffmpeg_available:
    st.error("⚠️ FFmpeg not found! Please install FFmpeg and add it to your PATH to export videos.")
    st.info("Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)")
from tracker.video_processor import VideoProcessor
from tracker.camera_path import CameraPath
from enhancement.quality_analyzer import QualityAnalyzer

# Cleanup at startup
if 'initialized' not in st.session_state:
    try:
        FileHandler.clear_temp("temp")
        FileHandler.clear_temp("outputs")
    except Exception:
        pass
    st.session_state.initialized = True

# Set page config
st.set_page_config(page_title="SravsDanceTrack AI", page_icon="💃", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
    }
    .stSidebar {
        background-color: #1a1c24;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'tracking_mode' not in st.session_state:
    st.session_state.tracking_mode = "AI Auto Tracking"
if 'keyframes' not in st.session_state:
    st.session_state.keyframes = []
if 'manual_bbox' not in st.session_state:
    st.session_state.manual_bbox = None
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None

# Sidebar - Settings
st.sidebar.title("⚙️ Settings")
tracking_mode = st.sidebar.selectbox("Tracking Mode", 
                                    ["AI Auto Tracking", "Manual Camera Tracking", "Manual Keyframe Camera Mode"])
st.session_state.tracking_mode = tracking_mode

st.sidebar.divider()
aspect_ratio = st.sidebar.selectbox("Output Aspect Ratio", ["9:16", "16:9", "1:1"])
zoom_factor = st.sidebar.slider("Zoom Factor", 1.0, 3.0, 1.2, 0.1)
smoothness = st.sidebar.slider("Smoothness", 0.0, 1.0, 0.85, 0.05)
smoothing_method = st.sidebar.selectbox("Smoothing Method", ["savgol", "moving_average", "exponential"])

if tracking_mode == "Manual Keyframe Camera Mode":
    interpolation = st.sidebar.selectbox("Interpolation", ["cubic", "linear", "bezier"])
else:
    interpolation = "linear"

st.sidebar.divider()
with st.sidebar.expander("✨ Enhanced Export Settings", expanded=False):
    preset = st.selectbox("Quality Preset", ["FAST", "HIGH QUALITY", "AI ENHANCED", "PREMIUM CINEMATIC"])
    
    # Defaults based on preset
    def_denoise = preset in ["HIGH QUALITY", "AI ENHANCED", "PREMIUM CINEMATIC"]
    def_sharpen = preset in ["HIGH QUALITY", "AI ENHANCED", "PREMIUM CINEMATIC"]
    def_color = preset in ["HIGH QUALITY", "AI ENHANCED", "PREMIUM CINEMATIC"]
    def_upscale = preset in ["AI ENHANCED", "PREMIUM CINEMATIC"]
    def_face = preset == "PREMIUM CINEMATIC"
    def_motion = preset == "PREMIUM CINEMATIC"
    def_codec = "H265" if preset == "PREMIUM CINEMATIC" else "H264"

    e_denoise = st.checkbox("Enable Denoising", value=def_denoise)
    e_sharpen = st.checkbox("Enable Sharpening", value=def_sharpen)
    e_color = st.checkbox("Enable Color Enhancement", value=def_color)
    e_upscale = st.checkbox("Enable AI Upscaling", value=def_upscale)
    e_face = st.checkbox("Enable Face Enhancement", value=def_face)
    e_motion = st.checkbox("Enable Motion Smoothing", value=def_motion)
    e_codec = st.radio("Export Codec", ["H264", "H265"], index=0 if def_codec == "H264" else 1)
    
    enhance_params = {
        "enable_denoising": e_denoise,
        "enable_sharpening": e_sharpen,
        "enable_color_enhancement": e_color,
        "enable_ai_upscaling": e_upscale,
        "enable_face_enhancement": e_face,
        "enable_motion_smoothing": e_motion,
        "export_codec": e_codec
    }

# Main UI
st.title("💃 SravsDanceTrack AI")
st.subheader("Keep your dancer centered with AI-powered virtual camera movement")

# Upload Section
uploaded_file = st.file_uploader("Upload Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save the file
    if st.session_state.video_path is None or uploaded_file.name != st.session_state.metadata.get('filename'):
        save_path = FileHandler.save_uploaded_file(uploaded_file)
        st.session_state.video_path = save_path
        st.session_state.metadata = FileHandler.get_video_metadata(save_path)
        st.session_state.keyframes = []
        st.session_state.manual_bbox = None
        st.session_state.processed_video = None
        st.rerun()

    meta = st.session_state.metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Resolution", meta['resolution'])
    col2.metric("FPS", f"{meta['fps']:.2f}")
    col3.metric("Duration", meta['duration'])

    # Quality Estimation Panel
    if meta:
        # Approximate current crop dims
        crop_h = meta['height'] / zoom_factor
        crop_w = crop_h * (720/1280 if aspect_ratio == "9:16" else 1280/720 if aspect_ratio == "16:9" else 1)
        target_w, target_h = (720, 1280) if aspect_ratio == "9:16" else (1280, 720) if aspect_ratio == "16:9" else (1080, 1080)
        
        quality = QualityAnalyzer.estimate_quality(crop_w, crop_h, target_w, target_h)
        with st.expander("📊 Estimated Output Quality", expanded=True):
            st.markdown(f"**Current Rating:** :{quality['color']}[{quality['rating']}]")
            st.write(f"**Zoom Level:** {quality['zoom_level']}")
            st.info(f"💡 {quality['suggestion']}")

    # Mode Specific UI
    if tracking_mode == "AI Auto Tracking":
        st.info("AI will automatically detect the dancer using Pose estimation.")
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("🚀 Standard Export"):
            processor = VideoProcessor(st.session_state.video_path)
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
            
            path_manager = processor.process_tracking(tracking_mode, progress_callback=update_progress)
            output_file = processor.generate_virtual_camera(
                path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                smoothness=smoothness, smoothing_method=smoothing_method, 
                progress_callback=update_progress
            )
            if output_file:
                st.session_state.processed_video = output_file
                st.success("✅ Video Processed (Standard)!")
                st.rerun()

        if col_btn2.button("✨ Enhanced Export"):
            processor = VideoProcessor(st.session_state.video_path)
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
            
            path_manager = processor.process_tracking(tracking_mode, progress_callback=update_progress)
            output_file = processor.generate_enhanced_video(
                path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                smoothness=smoothness, smoothing_method=smoothing_method, 
                enhance_settings=enhance_params,
                progress_callback=update_progress
            )
            if output_file:
                st.session_state.processed_video = output_file
                st.success("✅ Video Processed (Enhanced)!")
                st.rerun()

    elif tracking_mode == "Manual Camera Tracking":
        st.info("Select a target in the first frame to track.")
        
        # Load first frame
        cap = cv2.VideoCapture(st.session_state.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            st.write("### 🎯 Select Target to Track")
            st.info("Drag and resize the box over the subject you want to track.")
            
            # Using st_cropper for interactive bounding box selection
            # return_type='box' returns [left, top, width, height] in normalized or pixel coordinates
            box = st_cropper(img, realtime_update=True, box_color='#FF0000', return_type='box')
            
            if box:
                # Convert from st_cropper [left, top, width, height] to [x, y, w, h]
                bx, by, bw, bh = box['left'], box['top'], box['width'], box['height']
                st.write(f"Selected Area: X={bx}, Y={by}, W={bw}, H={bh}")
            
            col_m1, col_m2 = st.columns(2)
            if col_m1.button("🚀 Standard Export"):
                bbox = (int(bx), int(by), int(bw), int(bh))
                processor = VideoProcessor(st.session_state.video_path)
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
                path_manager = processor.process_tracking(tracking_mode, progress_callback=update_progress, bbox=bbox)
                output_file = processor.generate_virtual_camera(
                    path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                    smoothness=smoothness, smoothing_method=smoothing_method, 
                    progress_callback=update_progress
                )
                if output_file:
                    st.session_state.processed_video = output_file
                    st.success("✅ Video Processed (Standard)!")
                    st.rerun()

            if col_m2.button("✨ Enhanced Export"):
                bbox = (int(bx), int(by), int(bw), int(bh))
                processor = VideoProcessor(st.session_state.video_path)
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
                path_manager = processor.process_tracking(tracking_mode, progress_callback=update_progress, bbox=bbox)
                output_file = processor.generate_enhanced_video(
                    path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                    smoothness=smoothness, smoothing_method=smoothing_method, 
                    enhance_settings=enhance_params,
                    progress_callback=update_progress
                )
                if output_file:
                    st.session_state.processed_video = output_file
                    st.success("✅ Video Processed (Enhanced)!")
                    st.rerun()

    elif tracking_mode == "Manual Keyframe Camera Mode":
        st.info("Add keyframes at different time points to define the camera path.")
        
        # Timeline and Keyframe control
        current_frame_idx = st.slider("Timeline", 0, meta['frame_count']-1, 0)
        
        cap = cv2.VideoCapture(st.session_state.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            col_l, col_r = st.columns([2, 1])
            
            with col_l:
                st.write(f"Frame: {current_frame_idx}")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_container_width=True)
                
                kcol1, kcol2, kcol3 = st.columns(3)
                kx = kcol1.number_input("Target X", 0, meta['width'], meta['width']//2)
                ky = kcol2.number_input("Target Y", 0, meta['height'], meta['height']//2)
                if kcol3.button("➕ Add Keyframe"):
                    st.session_state.keyframes.append({"frame": current_frame_idx, "x": kx, "y": ky})
                    st.toast("Keyframe added!")
            
            with col_r:
                st.write("🔑 Keyframes")
                if st.session_state.keyframes:
                    for i, kf in enumerate(st.session_state.keyframes):
                        st.write(f"F:{kf['frame']} - ({kf['x']}, {kf['y']})")
                        if st.button(f"🗑️ Delete {i}", key=f"del_{i}"):
                            st.session_state.keyframes.pop(i)
                            st.rerun()
                else:
                    st.write("No keyframes added yet.")

        col_k1, col_k2 = st.columns(2)
        if col_k1.button("🚀 Standard Export"):
            if len(st.session_state.keyframes) < 2:
                st.error("Please add at least 2 keyframes.")
            else:
                processor = VideoProcessor(st.session_state.video_path)
                path_manager = CameraPath(meta['frame_count'])
                for kf in st.session_state.keyframes:
                    path_manager.add_keyframe(kf['frame'], kf['x'], kf['y'])
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
                
                output_file = processor.generate_virtual_camera(
                    path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                    smoothness=smoothness, smoothing_method=smoothing_method, 
                    interpolation_method=interpolation, is_keyframe_mode=True,
                    progress_callback=update_progress
                )
                if output_file:
                    st.session_state.processed_video = output_file
                    st.success("✅ Video Processed (Standard)!")
                    st.rerun()

        if col_k2.button("✨ Enhanced Export"):
            if len(st.session_state.keyframes) < 2:
                st.error("Please add at least 2 keyframes.")
            else:
                processor = VideoProcessor(st.session_state.video_path)
                path_manager = CameraPath(meta['frame_count'])
                for kf in st.session_state.keyframes:
                    path_manager.add_keyframe(kf['frame'], kf['x'], kf['y'])
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
                
                output_file = processor.generate_enhanced_video(
                    path_manager, aspect_ratio=aspect_ratio, zoom_factor=zoom_factor, 
                    smoothness=smoothness, 
                    smoothing_method=smoothing_method, 
                    enhance_settings=enhance_params,
                    interpolation_method=interpolation, is_keyframe_mode=True,
                    progress_callback=update_progress
                )
                if output_file:
                    st.session_state.processed_video = output_file
                    st.success("✅ Video Processed (Enhanced)!")
                    st.rerun()

# Preview Section
if st.session_state.processed_video:
    st.divider()
    st.subheader("📽️ Result Preview")
    
    col_orig, col_proc = st.columns(2)
    
    with col_orig:
        st.write("Original Video")
        st.video(st.session_state.video_path)
        
    with col_proc:
        st.write("Processed Video")
        st.video(st.session_state.processed_video)
        
    with open(st.session_state.processed_video, "rb") as f:
        st.download_button("📥 Download Exported Video", f, file_name="tracked_output.mp4")

# Footer
st.divider()
st.caption("Powered by DanceTrack AI - Advanced Motion Tracking")
