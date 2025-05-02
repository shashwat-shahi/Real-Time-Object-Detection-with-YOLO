#!/usr/bin/env python3
"""
Example script to demonstrate batch processing of multiple images
"""
import os
import sys
import glob
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.detector import ObjectDetector
from utils.helpers import setup_logging, is_image_file


def main():
    # Setup logging
    setup_logging("INFO")
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Example: Process all images in a directory
    input_dir = "data/images"  # Change this to your images directory
    output_dir = "data/results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Detect objects
            start_time = time.time()
            detections = detector.detect_objects(image_path)
            detection_time = time.time() - start_time
            
            # Draw and save annotated image
            annotated_img, _ = detector.detect_and_draw(image_path)
            
            output_filename = f"detected_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            
            import cv2
            cv2.imwrite(output_path, annotated_img)
            
            # Store results
            result = {
                "image": os.path.basename(image_path),
                "detections": len(detections),
                "time": detection_time,
                "objects": [
                    {
                        "class": d["class_name"],
                        "confidence": d["confidence"],
                        "bbox": d["bbox"]
                    }
                    for d in detections
                ]
            }
            results.append(result)
            
            print(f"  Found {len(detections)} objects in {detection_time:.3f}s")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    # Save results summary
    results_file = os.path.join(output_dir, "detection_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBatch processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {results_file}")
    
    # Print summary statistics
    total_objects = sum(r["detections"] for r in results)
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print(f"\nSummary:")
    print(f"  Images processed: {len(results)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Average processing time: {avg_time:.3f}s")


if __name__ == "__main__":
    main()