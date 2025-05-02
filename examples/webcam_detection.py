#!/usr/bin/env python3
"""
Real-time webcam detection example
"""
import argparse
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.detector import ObjectDetector
from utils.helpers import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Real-time webcam detection')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('-m', '--model', default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('-o', '--output', help='Output video file path (optional)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize detector
    try:
        detector = ObjectDetector()
        detector.switch_model(args.model)
        logger.info(f"Loaded model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Start detection
    try:
        logger.info(f"Starting webcam detection (camera {args.camera})")
        logger.info("Press 'q' to quit")
        
        detector.detect_video_stream(
            source=args.camera,
            output_path=args.output,
            conf_threshold=args.confidence,
            display=True
        )
        
    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())