"""
Flask API for YOLO Object Detection
"""
import os
import logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile

try:
    from ..detection.detector import ObjectDetector
    from ..utils.helpers import setup_logging, validate_file_upload, format_detection_results
except ImportError:
    from detection.detector import ObjectDetector
    from utils.helpers import setup_logging, validate_file_upload, format_detection_results

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize detector
detector = ObjectDetector()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """API index endpoint"""
    return jsonify({
        "message": "YOLO Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect/upload": "POST - Upload image/video for detection",
            "/detect/url": "POST - Detect objects from image URL",
            "/detect/base64": "POST - Detect objects from base64 image",
            "/models": "GET - List available models",
            "/models/current": "GET - Get current model info",
            "/models/switch": "POST - Switch to different model"
        }
    })


@app.route('/detect/upload', methods=['POST'])
def detect_upload():
    """Detect objects in uploaded file"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not supported"}), 400
        
        # Get parameters
        conf_threshold = request.form.get('confidence', type=float)
        iou_threshold = request.form.get('iou', type=float)
        draw_boxes = request.form.get('draw_boxes', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Validate file
        is_valid, error_msg = validate_file_upload(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({"error": error_msg}), 400
        
        # Detect objects
        detections = detector.detect_objects(
            filepath, 
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        result = format_detection_results(detections)
        result["filename"] = filename
        result["model"] = detector.get_model_info()
        
        # If draw_boxes is True, return annotated image
        if draw_boxes:
            annotated_img, _ = detector.detect_and_draw(filepath, conf_threshold, iou_threshold)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/detect/base64', methods=['POST'])
def detect_base64():
    """Detect objects in base64 encoded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get parameters
        conf_threshold = data.get('confidence')
        iou_threshold = data.get('iou')
        draw_boxes = data.get('draw_boxes', True)
        
        # Detect objects
        detections = detector.detect_objects(
            img_array,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        result = format_detection_results(detections)
        result["model"] = detector.get_model_info()
        
        # If draw_boxes is True, return annotated image
        if draw_boxes:
            annotated_img, _ = detector.detect_and_draw(img_array, conf_threshold, iou_threshold)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        available_models = detector.model_manager.get_available_models()
        current_model = detector.model_manager.get_current_model_name()
        
        return jsonify({
            "available_models": available_models,
            "current_model": current_model
        })
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/models/current', methods=['GET'])
def get_current_model():
    """Get current model information"""
    try:
        model_info = detector.get_model_info()
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({"error": "Model name not provided"}), 400
        
        model_name = data['model_name']
        available_models = detector.model_manager.get_available_models()
        
        if model_name not in available_models:
            return jsonify({"error": f"Model {model_name} not available"}), 400
        
        detector.switch_model(model_name)
        model_info = detector.get_model_info()
        
        return jsonify({
            "message": f"Switched to model: {model_name}",
            "model_info": model_info
        })
        
    except Exception as e:
        logger.error(f"Failed to switch model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large"}), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)