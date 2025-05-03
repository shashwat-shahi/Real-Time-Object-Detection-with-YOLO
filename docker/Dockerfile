FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models uploads data/images data/results

# Expose ports
EXPOSE 5000 8000 8501

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FLASK_APP=src/api/flask_app.py
ENV FASTAPI_APP=src/api/fastapi_app.py

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]