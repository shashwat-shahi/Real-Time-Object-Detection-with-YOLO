"""
Utility functions for object detection system
"""
import os
import logging
import yaml
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
from PIL import Image


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def is_image_file(file_path: str) -> bool:
    """Check if file is a supported image format"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return Path(file_path).suffix.lower() in image_extensions


def is_video_file(file_path: str) -> bool:
    """Check if file is a supported video format"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    return Path(file_path).suffix.lower() in video_extensions


def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img


def save_image(image: np.ndarray, output_path: str):
    """Save image to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def resize_image(image: np.ndarray, target_size: tuple, maintain_aspect: bool = True) -> np.ndarray:
    """Resize image while optionally maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if maintain_aspect:
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, target_size)


def create_directories(directories: List[str]):
    """Create directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)


def validate_file_upload(file_path: str, max_size_mb: float = 50) -> tuple:
    """
    Validate uploaded file
    
    Returns:
        (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size
    file_size = get_file_size_mb(file_path)
    if file_size > max_size_mb:
        return False, f"File too large: {file_size:.1f}MB (max: {max_size_mb}MB)"
    
    # Check file type
    if not (is_image_file(file_path) or is_video_file(file_path)):
        return False, "Unsupported file format"
    
    return True, "Valid file"


def format_detection_results(detections: List[dict]) -> dict:
    """Format detection results for API response"""
    return {
        "total_objects": len(detections),
        "detections": detections,
        "classes_detected": list(set(d["class_name"] for d in detections)),
        "class_counts": {
            class_name: sum(1 for d in detections if d["class_name"] == class_name)
            for class_name in set(d["class_name"] for d in detections)
        }
    }


def convert_pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format"""
    # Convert PIL to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    cv2_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    
    return cv2_image


def convert_cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image