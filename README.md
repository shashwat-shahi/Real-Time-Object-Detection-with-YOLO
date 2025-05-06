# 🔍 Real-Time Object Detection with YOLO

A comprehensive real-time object detection system built with YOLO (You Only Look Once) architecture. This system supports multiple YOLO models, provides web interfaces, REST APIs, and can detect objects in images, videos, and live webcam streams.

## ✨ Features

- **🎯 Multiple YOLO Models**: YOLOv8 (all variants), YOLOv7, YOLO-NAS support
- **🌐 Web Interfaces**: Streamlit dashboard, Flask and FastAPI REST APIs
- **📷 Multiple Input Types**: Images, videos, webcam streams, batch processing
- **🐳 Docker Ready**: Complete containerization with docker-compose
- **📊 Rich Analytics**: Detection statistics, class distribution, performance metrics
- **🔧 Configurable**: Easily adjustable confidence thresholds, IoU settings
- **📱 User Friendly**: Intuitive web interface for non-technical users

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Clone the repository
git clone https://github.com/shashwat-shahi/Real-Time-Object-Detection-with-YOLO.git
cd Real-Time-Object-Detection-with-YOLO

# Run quick start script
./quick_start.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs models uploads data/{images,results}
```

## 📖 Usage

### 🖼️ Image Detection
```bash
# Detect objects in a single image
python examples/detect_cli.py path/to/image.jpg

# With custom confidence threshold
python examples/detect_cli.py path/to/image.jpg --confidence 0.5

# Save annotated image
python examples/detect_cli.py path/to/image.jpg -o output/detected_image.jpg
```

### 📹 Video Processing
```bash
# Process video file
python examples/detect_cli.py path/to/video.mp4 -o output/detected_video.mp4

# Real-time webcam detection
python examples/webcam_detection.py
```

### 🌐 Web Interfaces

#### Streamlit Dashboard (Recommended for beginners)
```bash
streamlit run src/web/streamlit_app.py
```
Access at: http://localhost:8501

#### FastAPI (For developers)
```bash
uvicorn src.api.fastapi_app:app --reload
```
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

#### Flask API (Alternative)
```bash
python src/api/flask_app.py
```
Access at: http://localhost:5000

### 🐳 Docker Deployment

```bash
# FastAPI service
docker-compose -f docker/docker-compose.yml up yolo-api

# Streamlit service
docker-compose -f docker/docker-compose.yml --profile streamlit up

# All services
docker-compose -f docker/docker-compose.yml --profile streamlit --profile flask up
```

## 🏗️ Architecture

```
Real-Time-Object-Detection-with-YOLO/
├── src/
│   ├── models/          # Model management
│   ├── detection/       # Core detection engine
│   ├── api/            # REST APIs (Flask, FastAPI)
│   ├── web/            # Streamlit interface
│   └── utils/          # Utilities and helpers
├── examples/           # Example scripts
├── tests/              # Unit tests
├── config/             # Configuration files
├── docker/             # Docker configuration
└── data/               # Data directories
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  default: "yolov8n.pt"  # Default model
  confidence_threshold: 0.25
  iou_threshold: 0.45

detection:
  input_size: 640
  max_det: 1000

api:
  host: "0.0.0.0"
  port: 8000
```

## 📊 Supported Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.2MB | Fastest | Good | Real-time applications |
| YOLOv8s | 21.5MB | Fast | Better | Balanced performance |
| YOLOv8m | 49.7MB | Medium | High | High accuracy needs |
| YOLOv8l | 83.7MB | Slow | Higher | Production systems |
| YOLOv8x | 136.7MB | Slowest | Highest | Maximum accuracy |

## 🗂️ Supported Datasets

- **COCO Dataset**: 80 object classes, 330k images
- **Open Images V7**: 600+ classes, millions of images  
- **Pascal VOC**: 20 object classes
- **Custom datasets**: Support for custom trained models

## 📡 API Endpoints

### FastAPI/Flask Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/detect/upload` | POST | Upload image/video for detection |
| `/detect/base64` | POST | Detect from base64 image |
| `/models` | GET | List available models |
| `/models/current` | GET | Current model info |
| `/models/switch` | POST | Switch model |

### Example API Usage

```python
import requests

# Upload image for detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/upload',
        files={'file': f},
        data={'confidence': 0.5}
    )
    
results = response.json()
print(f"Detected {results['total_objects']} objects")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src tests/
```

## 📦 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Computer vision library
- **FastAPI/Flask**: Web frameworks
- **Streamlit**: Web dashboard

### Optional Dependencies
- **Roboflow**: Dataset management
- **LabelImg**: Image annotation
- **Docker**: Containerization

## 🚀 Performance

### Benchmarks (on CPU)
- **YOLOv8n**: ~50 FPS (640x640)
- **YOLOv8s**: ~35 FPS (640x640)
- **YOLOv8m**: ~25 FPS (640x640)

*Note: Performance varies based on hardware and image resolution*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8 implementation
- [COCO Dataset](https://cocodataset.org/) for training data
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Support

- 📧 **Issues**: Use GitHub Issues for bug reports
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📖 **Documentation**: Check the `/docs` folder for detailed guides

## 🎯 Roadmap

- [ ] Mobile app integration
- [ ] Edge deployment (ONNX, TensorRT)
- [ ] Custom model training pipeline
- [ ] Multi-camera support
- [ ] Object tracking capabilities
- [ ] Integration with cloud storage

---

**Made with ❤️ for the Computer Vision Community**
