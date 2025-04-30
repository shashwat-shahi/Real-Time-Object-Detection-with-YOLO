"""
Object Detection Engine - Core detection functionality
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import time
from PIL import Image

try:
    from ..models.model_manager import YOLOModelManager
except ImportError:
    from models.model_manager import YOLOModelManager

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Core object detection engine using YOLO models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the object detector"""
        self.model_manager = YOLOModelManager(config_path)
        self.config = self.model_manager.config
        
    def detect_objects(self, 
                      image: Union[str, np.ndarray, Image.Image],
                      conf_threshold: Optional[float] = None,
                      iou_threshold: Optional[float] = None,
                      max_det: Optional[int] = None) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            conf_threshold: Confidence threshold (overrides config)
            iou_threshold: IoU threshold (overrides config)
            max_det: Maximum detections (overrides config)
            
        Returns:
            List of detection dictionaries
        """
        if self.model_manager.current_model is None:
            raise ValueError("No model loaded")
        
        # Set parameters
        conf = conf_threshold or self.config['model']['confidence_threshold']
        iou = iou_threshold or self.config['model']['iou_threshold']
        max_detections = max_det or self.config['detection']['max_det']
        
        try:
            # Perform inference
            start_time = time.time()
            results = self.model_manager.current_model(
                image,
                conf=conf,
                iou=iou,
                max_det=max_detections,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First image result
                
                if result.boxes is not None:
                    boxes = result.boxes
                    class_names = self.model_manager.get_class_names()
                    
                    for i in range(len(boxes)):
                        # Extract box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            "bbox": {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                                "width": float(box[2] - box[0]),
                                "height": float(box[3] - box[1])
                            },
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name
                        }
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects in {inference_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise
    
    def detect_and_draw(self, 
                       image: Union[str, np.ndarray],
                       conf_threshold: Optional[float] = None,
                       iou_threshold: Optional[float] = None,
                       draw_labels: bool = True,
                       line_thickness: int = 2) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects and draw bounding boxes on image
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            draw_labels: Whether to draw class labels
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Tuple of (annotated_image, detections)
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()
        
        # Detect objects
        detections = self.detect_objects(image, conf_threshold, iou_threshold)
        
        # Draw bounding boxes
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # Draw rectangle
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # Color based on class (simple hash-based coloring)
            color = self._get_color_for_class(detection["class_id"])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            if draw_labels:
                # Draw label background
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 1)
        
        return img, detections
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color for a class ID"""
        # Simple hash-based color generation
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def detect_video_stream(self, 
                           source: Union[str, int] = 0,
                           output_path: Optional[str] = None,
                           conf_threshold: Optional[float] = None,
                           display: bool = True) -> None:
        """
        Detect objects in video stream (webcam or video file)
        
        Args:
            source: Video source (0 for webcam, path for video file)
            output_path: Path to save output video
            conf_threshold: Confidence threshold
            display: Whether to display video in window
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and draw
                annotated_frame, detections = self.detect_and_draw(
                    frame, conf_threshold=conf_threshold
                )
                
                # Add FPS counter
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save frame if writer initialized
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('YOLO Object Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return self.model_manager.get_model_info()
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different YOLO model"""
        self.model_manager.load_model(model_name)
        logger.info(f"Switched to model: {model_name}")