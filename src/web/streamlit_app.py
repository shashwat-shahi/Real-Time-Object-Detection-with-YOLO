"""
Streamlit Web Interface for YOLO Object Detection
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import logging
from io import BytesIO
import time

# Set up the page config
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import detection modules
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detection.detector import ObjectDetector
    from utils.helpers import setup_logging, format_detection_results
except ImportError as e:
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from detection.detector import ObjectDetector
        from utils.helpers import setup_logging, format_detection_results
    except ImportError:
        st.error(f"Failed to import detection modules: {e}")
        st.stop()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize session state
if 'detector' not in st.session_state:
    try:
        st.session_state.detector = ObjectDetector()
        st.session_state.detection_history = []
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        st.stop()


def main():
    """Main Streamlit application"""
    st.title("🔍 Real-Time Object Detection with YOLO")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = st.session_state.detector.model_manager.get_available_models()
        current_model = st.session_state.detector.model_manager.get_current_model_name()
        
        selected_model = st.selectbox(
            "Select YOLO Model:",
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        
        if selected_model != current_model:
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    st.session_state.detector.switch_model(selected_model)
                    st.success(f"Switched to {selected_model}")
                except Exception as e:
                    st.error(f"Failed to switch model: {e}")
        
        st.markdown("---")
        
        # Detection parameters
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        draw_labels = st.checkbox("Draw Labels", value=True)
        line_thickness = st.slider("Line Thickness", 1, 5, 2)
        
        st.markdown("---")
        
        # Model information
        st.subheader("📊 Model Information")
        model_info = st.session_state.detector.get_model_info()
        st.json(model_info)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📹 Webcam", "📈 History"])
    
    with tab1:
        image_detection_tab(confidence_threshold, iou_threshold, draw_labels, line_thickness)
    
    with tab2:
        video_detection_tab(confidence_threshold, iou_threshold, draw_labels, line_thickness)
    
    with tab3:
        webcam_detection_tab(confidence_threshold, iou_threshold, draw_labels, line_thickness)
    
    with tab4:
        history_tab()


def image_detection_tab(conf_threshold, iou_threshold, draw_labels, line_thickness):
    """Image detection tab"""
    st.header("📷 Image Object Detection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        help="Upload an image file for object detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform detection
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Detecting objects..."):
                try:
                    # Convert PIL to OpenCV format
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Detect objects
                    start_time = time.time()
                    detections = st.session_state.detector.detect_objects(
                        img_array,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold
                    )
                    detection_time = time.time() - start_time
                    
                    # Draw bounding boxes
                    annotated_img, _ = st.session_state.detector.detect_and_draw(
                        img_array,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        draw_labels=draw_labels,
                        line_thickness=line_thickness
                    )
                    
                    # Convert back to RGB for display
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_img, caption="Detected Objects", use_column_width=True)
                    
                    # Display results
                    st.markdown("---")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.metric("Objects Detected", len(detections))
                        st.metric("Detection Time", f"{detection_time:.3f}s")
                    
                    with col4:
                        if detections:
                            classes_detected = list(set(d["class_name"] for d in detections))
                            st.metric("Unique Classes", len(classes_detected))
                            
                            # Class counts
                            class_counts = {}
                            for d in detections:
                                class_name = d["class_name"]
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            st.subheader("Class Distribution")
                            st.bar_chart(class_counts)
                    
                    # Detailed results
                    if detections:
                        st.subheader("Detailed Detection Results")
                        
                        for i, detection in enumerate(detections):
                            with st.expander(f"Detection {i+1}: {detection['class_name']} ({detection['confidence']:.2f})"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Class:** {detection['class_name']}")
                                    st.write(f"**Confidence:** {detection['confidence']:.3f}")
                                with col_b:
                                    bbox = detection['bbox']
                                    st.write(f"**Position:** ({bbox['x1']:.0f}, {bbox['y1']:.0f})")
                                    st.write(f"**Size:** {bbox['width']:.0f} x {bbox['height']:.0f}")
                    
                    # Add to history
                    st.session_state.detection_history.append({
                        "timestamp": time.time(),
                        "type": "image",
                        "filename": uploaded_file.name,
                        "detections": len(detections),
                        "time": detection_time,
                        "classes": list(set(d["class_name"] for d in detections))
                    })
                    
                except Exception as e:
                    st.error(f"Detection failed: {e}")


def video_detection_tab(conf_threshold, iou_threshold, draw_labels, line_thickness):
    """Video detection tab"""
    st.header("🎥 Video Object Detection")
    
    # File uploader
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for object detection"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        st.info(f"Video: {uploaded_video.name} | Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {frame_count}")
        
        # Process video
        if st.button("🔍 Process Video", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create temporary output file
                output_path = tempfile.mktemp(suffix="_detected.mp4")
                
                # Process video
                cap = cv2.VideoCapture(temp_video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_idx = 0
                total_detections = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect objects in frame
                    detections = st.session_state.detector.detect_objects(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold
                    )
                    
                    # Draw bounding boxes
                    annotated_frame, _ = st.session_state.detector.detect_and_draw(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        draw_labels=draw_labels,
                        line_thickness=line_thickness
                    )
                    
                    out.write(annotated_frame)
                    total_detections += len(detections)
                    
                    # Update progress
                    frame_idx += 1
                    progress = frame_idx / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_idx}/{frame_count} | Objects: {len(detections)}")
                
                cap.release()
                out.release()
                
                # Display results
                st.success(f"Video processing complete! Total objects detected: {total_detections}")
                
                # Provide download link
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=file.read(),
                        file_name=f"detected_{uploaded_video.name}",
                        mime="video/mp4"
                    )
                
                # Clean up
                os.unlink(output_path)
                
            except Exception as e:
                st.error(f"Video processing failed: {e}")
            finally:
                # Clean up temp file
                os.unlink(temp_video_path)


def webcam_detection_tab(conf_threshold, iou_threshold, draw_labels, line_thickness):
    """Webcam detection tab"""
    st.header("📹 Real-Time Webcam Detection")
    
    st.info("🚧 Webcam detection requires additional setup for Streamlit deployment. This feature works best in local environments.")
    
    # Placeholder for webcam implementation
    st.markdown("""
    ### To enable webcam detection:
    
    1. **Local Development**: Use the example scripts provided
    2. **Production**: Consider using WebRTC or similar technologies
    3. **Alternative**: Upload images from your device camera
    
    For now, you can use the image detection tab with photos taken from your camera.
    """)
    
    # Simple webcam simulation with uploaded images
    if st.button("📸 Take Photo (Upload from Camera)"):
        st.info("Please use your device's camera app and upload the image using the Image Detection tab.")


def history_tab():
    """Detection history tab"""
    st.header("📈 Detection History")
    
    if st.session_state.detection_history:
        # Convert to DataFrame for display
        import pandas as pd
        
        history_data = []
        for item in st.session_state.detection_history:
            history_data.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["timestamp"])),
                "Type": item["type"],
                "Filename": item["filename"],
                "Objects Detected": item["detections"],
                "Processing Time (s)": f"{item['time']:.3f}",
                "Classes": ", ".join(item["classes"][:3]) + ("..." if len(item["classes"]) > 3 else "")
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_files = len(st.session_state.detection_history)
            st.metric("Total Files Processed", total_files)
        
        with col2:
            total_objects = sum(item["detections"] for item in st.session_state.detection_history)
            st.metric("Total Objects Detected", total_objects)
        
        with col3:
            avg_time = sum(item["time"] for item in st.session_state.detection_history) / len(st.session_state.detection_history)
            st.metric("Average Processing Time", f"{avg_time:.3f}s")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    
    else:
        st.info("No detection history available. Process some images or videos to see history here.")


if __name__ == "__main__":
    main()