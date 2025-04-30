"""
FastAPI implementation for YOLO Object Detection
"""
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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

app = FastAPI(
    title="YOLO Object Detection API",
    description="Real-time object detection using YOLO architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = ObjectDetector()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Pydantic models
class Base64ImageRequest(BaseModel):
    image: str
    confidence: Optional[float] = None
    iou: Optional[float] = None
    draw_boxes: Optional[bool] = True


class ModelSwitchRequest(BaseModel):
    model_name: str


class DetectionResponse(BaseModel):
    total_objects: int
    detections: List[dict]
    classes_detected: List[str]
    class_counts: dict
    model: dict
    annotated_image: Optional[str] = None


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "YOLO Object Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "/detect/upload": "POST - Upload image/video for detection",
            "/detect/base64": "POST - Detect objects from base64 image",
            "/models": "GET - List available models",
            "/models/current": "GET - Get current model info",
            "/models/switch": "POST - Switch to different model"
        }
    }


@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_upload(
    file: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    draw_boxes: Optional[bool] = Form(True)
):
    """Detect objects in uploaded file"""
    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Read file content
        content = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Detect objects
        detections = detector.detect_objects(
            img,
            conf_threshold=confidence,
            iou_threshold=iou
        )
        
        result = format_detection_results(detections)
        result["model"] = detector.get_model_info()
        
        # If draw_boxes is True, return annotated image
        if draw_boxes:
            annotated_img, _ = detector.detect_and_draw(img, confidence, iou)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_base64(request: Base64ImageRequest):
    """Detect objects in base64 encoded image"""
    try:
        # Decode base64 image
        image_data = request.image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect objects
        detections = detector.detect_objects(
            img_array,
            conf_threshold=request.confidence,
            iou_threshold=request.iou
        )
        
        result = format_detection_results(detections)
        result["model"] = detector.get_model_info()
        
        # If draw_boxes is True, return annotated image
        if request.draw_boxes:
            annotated_img, _ = detector.detect_and_draw(
                img_array, request.confidence, request.iou
            )
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models"""
    try:
        available_models = detector.model_manager.get_available_models()
        current_model = detector.model_manager.get_current_model_name()
        
        return {
            "available_models": available_models,
            "current_model": current_model
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/current")
async def get_current_model():
    """Get current model information"""
    try:
        model_info = detector.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        model_name = request.model_name
        available_models = detector.model_manager.get_available_models()
        
        if model_name not in available_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {model_name} not available"
            )
        
        detector.switch_model(model_name)
        model_info = detector.get_model_info()
        
        return {
            "message": f"Switched to model: {model_name}",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Failed to switch model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)