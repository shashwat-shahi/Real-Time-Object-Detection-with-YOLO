"""
Unit tests for YOLO model manager
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_manager import YOLOModelManager


class TestYOLOModelManager:
    """Test cases for YOLOModelManager"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary config
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
        config_data = {
            'model': {
                'default': 'yolov8n.pt',
                'available_models': ['yolov8n.pt', 'yolov8s.pt'],
                'model_dir': 'test_models/',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'detection': {
                'input_size': 640,
                'max_det': 1000
            }
        }
        yaml.dump(config_data, self.temp_config)
        self.temp_config.close()
    
    def teardown_method(self):
        """Clean up test environment"""
        os.unlink(self.temp_config.name)
    
    def test_load_config(self):
        """Test configuration loading"""
        manager = YOLOModelManager(self.temp_config.name)
        assert manager.config['model']['default'] == 'yolov8n.pt'
        assert manager.config['model']['confidence_threshold'] == 0.25
    
    def test_get_available_models(self):
        """Test getting available models"""
        manager = YOLOModelManager(self.temp_config.name)
        models = manager.get_available_models()
        assert 'yolov8n.pt' in models
        assert 'yolov8s.pt' in models
    
    def test_default_config(self):
        """Test default configuration when file not found"""
        manager = YOLOModelManager("nonexistent_config.yaml")
        config = manager.config
        assert 'model' in config
        assert 'detection' in config
    
    @patch('models.model_manager.YOLO')
    def test_load_model(self, mock_yolo):
        """Test model loading"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        manager = YOLOModelManager(self.temp_config.name)
        model = manager.load_model('yolov8n.pt')
        
        assert model == mock_model
        assert manager.current_model == mock_model
        assert manager.current_model_name == 'yolov8n.pt'
    
    def test_get_model_info(self):
        """Test getting model information"""
        with patch('models.model_manager.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.names = {0: 'person', 1: 'car'}
            mock_yolo.return_value = mock_model
            
            manager = YOLOModelManager(self.temp_config.name)
            info = manager.get_model_info()
            
            assert 'name' in info
            assert 'classes' in info


if __name__ == '__main__':
    pytest.main([__file__])