"""
Unit tests for object detector
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.detector import ObjectDetector


class TestObjectDetector:
    """Test cases for ObjectDetector"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create a simple test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Save test image
        self.temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(self.temp_image.name, self.test_image)
        self.temp_image.close()
    
    def teardown_method(self):
        """Clean up test environment"""
        os.unlink(self.temp_image.name)
    
    @patch('detection.detector.YOLOModelManager')
    def test_detector_initialization(self, mock_manager_class):
        """Test detector initialization"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        detector = ObjectDetector()
        assert detector.model_manager == mock_manager
    
    @patch('detection.detector.YOLOModelManager')
    def test_detect_objects(self, mock_manager_class):
        """Test object detection"""
        # Setup mocks
        mock_manager = Mock()
        mock_model = Mock()
        mock_manager.current_model = mock_model
        mock_manager.get_class_names.return_value = {0: 'person', 1: 'car'}
        mock_manager_class.return_value = mock_manager
        
        # Mock detection results
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
        mock_boxes.conf = [np.array(0.85)]
        mock_boxes.cls = [np.array(0)]
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        detector = ObjectDetector()
        detections = detector.detect_objects(self.test_image)
        
        assert len(detections) == 1
        assert detections[0]['class_name'] == 'person'
        assert detections[0]['confidence'] == 0.85
        assert 'bbox' in detections[0]
    
    @patch('detection.detector.YOLOModelManager')
    def test_detect_and_draw(self, mock_manager_class):
        """Test detection with drawing"""
        # Setup mocks
        mock_manager = Mock()
        mock_model = Mock()
        mock_manager.current_model = mock_model
        mock_manager.get_class_names.return_value = {0: 'person'}
        mock_manager_class.return_value = mock_manager
        
        # Mock detection results
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy = [np.array([100, 100, 200, 200])]
        mock_boxes.conf = [np.array(0.85)]
        mock_boxes.cls = [np.array(0)]
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        detector = ObjectDetector()
        annotated_img, detections = detector.detect_and_draw(self.test_image)
        
        assert annotated_img.shape == self.test_image.shape
        assert len(detections) == 1
    
    def test_get_color_for_class(self):
        """Test color generation for classes"""
        with patch('detection.detector.YOLOModelManager'):
            detector = ObjectDetector()
            
            color1 = detector._get_color_for_class(0)
            color2 = detector._get_color_for_class(1)
            color3 = detector._get_color_for_class(0)  # Same class
            
            assert isinstance(color1, tuple)
            assert len(color1) == 3
            assert color1 == color3  # Same class should give same color
            assert color1 != color2  # Different classes should give different colors


if __name__ == '__main__':
    pytest.main([__file__])