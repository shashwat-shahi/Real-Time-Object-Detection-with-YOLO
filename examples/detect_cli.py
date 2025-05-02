#!/usr/bin/env python3
"""
Command-line interface for YOLO object detection
"""
import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.detector import ObjectDetector
from utils.helpers import setup_logging, is_image_file, is_video_file


def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection CLI')
    parser.add_argument('input', help='Input image or video file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-m', '--model', default='yolov8n.pt', 
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold (default: 0.45)')
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t display output (save only)')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save detection results as text file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Initialize detector
    try:
        detector = ObjectDetector()
        detector.switch_model(args.model)
        logger.info(f"Loaded model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Process input
    try:
        if is_image_file(args.input):
            process_image(detector, args, logger)
        elif is_video_file(args.input):
            process_video(detector, args, logger)
        else:
            logger.error("Unsupported file format")
            return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


def process_image(detector, args, logger):
    """Process a single image"""
    logger.info(f"Processing image: {args.input}")
    
    # Detect objects
    detections = detector.detect_objects(
        args.input,
        conf_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    logger.info(f"Detected {len(detections)} objects")
    
    # Print detection results
    for i, detection in enumerate(detections, 1):
        bbox = detection['bbox']
        logger.info(f"  {i}. {detection['class_name']} "
                   f"(conf: {detection['confidence']:.3f}) "
                   f"at [{bbox['x1']:.0f}, {bbox['y1']:.0f}, "
                   f"{bbox['x2']:.0f}, {bbox['y2']:.0f}]")
    
    # Save annotated image if output specified
    if args.output or not args.no_display:
        annotated_img, _ = detector.detect_and_draw(
            args.input,
            conf_threshold=args.confidence,
            iou_threshold=args.iou
        )
        
        if args.output:
            import cv2
            output_path = args.output
            cv2.imwrite(output_path, annotated_img)
            logger.info(f"Saved annotated image to: {output_path}")
        
        if not args.no_display:
            import cv2
            cv2.imshow('YOLO Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Save text results if requested
    if args.save_txt:
        txt_path = Path(args.input).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                f.write(f"{detection['class_name']} {detection['confidence']:.6f} "
                       f"{bbox['x1']:.2f} {bbox['y1']:.2f} "
                       f"{bbox['x2']:.2f} {bbox['y2']:.2f}\n")
        logger.info(f"Saved detection results to: {txt_path}")


def process_video(detector, args, logger):
    """Process a video file"""
    logger.info(f"Processing video: {args.input}")
    
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = input_path.with_stem(f"{input_path.stem}_detected")
    
    # Process video
    detector.detect_video_stream(
        source=args.input,
        output_path=output_path,
        conf_threshold=args.confidence,
        display=not args.no_display
    )
    
    logger.info(f"Processed video saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())