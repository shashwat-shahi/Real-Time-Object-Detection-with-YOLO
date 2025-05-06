#!/bin/bash

# Real-Time Object Detection with YOLO - Quick Start Script

echo "🚀 YOLO Object Detection - Quick Start"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models uploads data/{images,results}

echo ""
echo "✅ Setup complete! You can now:"
echo ""
echo "🖼️  Process images:"
echo "   python examples/detect_cli.py path/to/image.jpg"
echo ""
echo "📹 Start webcam detection:"
echo "   python examples/webcam_detection.py"
echo ""
echo "🌐 Launch web interfaces:"
echo "   # Streamlit (recommended for beginners)"
echo "   streamlit run src/web/streamlit_app.py"
echo ""
echo "   # FastAPI (for developers)"
echo "   uvicorn src.api.fastapi_app:app --reload"
echo ""
echo "   # Flask (alternative API)"
echo "   python src/api/flask_app.py"
echo ""
echo "🐳 Docker deployment:"
echo "   docker-compose -f docker/docker-compose.yml up"
echo ""
echo "📚 For more information, see README.md"