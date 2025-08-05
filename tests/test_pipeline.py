"""
Test suite for GIS Image Analysis pipeline
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.pipeline import GISAnalyzer
from src.utils.config import load_config
from src.preprocessing.image_processor import ImageProcessor
from src.detection.yolo_detector import YOLODetector
from src.segmentation.deeplabv3_segmenter import DeepLabV3Segmenter
from src.evaluation.metrics import MetricsCalculator
from src.visualization.visualizer import Visualizer


class TestImageProcessor:
    """Test image preprocessing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "target_size": (512, 512),
            "normalize": True,
            "enhance_contrast": True
        }
        self.device = torch.device("cpu")
        self.processor = ImageProcessor(self.config, self.device)
        
    def test_rgb_extraction(self):
        """Test RGB band extraction."""
        # Test with 3-band image
        image = np.random.rand(3, 256, 256).astype(np.float32)
        rgb = self.processor._extract_rgb_bands(image, {})
        assert rgb.shape == (256, 256, 3)
        
        # Test with single band
        image = np.random.rand(1, 256, 256).astype(np.float32)
        rgb = self.processor._extract_rgb_bands(image, {})
        assert rgb.shape == (256, 256, 3)
        
    def test_normalization(self):
        """Test image normalization."""
        image = np.random.rand(256, 256, 3) * 255
        normalized = self.processor._normalize_image(image)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
    def test_resize(self):
        """Test image resizing."""
        image = np.random.rand(128, 128, 3)
        resized = self.processor._resize_image(image, (256, 256))
        assert resized.shape == (256, 256, 3)


class TestYOLODetector:
    """Test YOLO object detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "model_size": "n",  # Use nano for testing
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5
        }
        self.models_dir = Path("models")
        self.device = torch.device("cpu")
        
    def test_iou_calculation(self):
        """Test IoU calculation."""
        # Create mock detector without loading model
        detector = YOLODetector.__new__(YOLODetector)
        detector.config = self.config
        detector.iou_threshold = 0.5
        
        # Test boxes
        boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        boxes2 = torch.tensor([[0, 0, 10, 10], [12, 12, 22, 22]])
        
        iou = detector.calculate_iou(boxes1, boxes2)
        assert iou.shape == (2, 2)
        assert iou[0, 0] == 1.0  # Perfect overlap
        
    def test_detection_summary(self):
        """Test detection summary creation."""
        detector = YOLODetector.__new__(YOLODetector)
        
        class_names = ["building", "vehicle", "building"]
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        summary = detector._create_detection_summary(class_names, scores)
        assert summary["total_detections"] == 3
        assert summary["unique_classes"] == 2
        assert "building" in summary["class_distribution"]


class TestDeepLabV3Segmenter:
    """Test DeepLabV3+ segmentation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "num_classes": 6,
            "encoder_name": "resnet18",  # Use smaller model for testing
            "encoder_weights": None
        }
        self.models_dir = Path("models")
        self.device = torch.device("cpu")
        
    def test_class_statistics(self):
        """Test class statistics calculation."""
        segmenter = DeepLabV3Segmenter.__new__(DeepLabV3Segmenter)
        segmenter.class_names = ["bg", "vegetation", "urban"]
        
        predictions = torch.tensor([[0, 1, 2], [1, 1, 2], [0, 2, 2]])
        probabilities = torch.rand(3, 3, 3)
        
        stats = segmenter._calculate_class_statistics(predictions, probabilities)
        assert len(stats) == 3
        assert all(key in stats for key in segmenter.class_names)
        
    def test_colored_mask_creation(self):
        """Test colored mask creation."""
        segmenter = DeepLabV3Segmenter.__new__(DeepLabV3Segmenter)
        segmenter.class_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        
        predictions = torch.tensor([[0, 1], [2, 1]])
        colored_mask = segmenter._create_colored_mask(predictions)
        
        assert colored_mask.shape == (2, 2, 3)
        assert np.array_equal(colored_mask[0, 0], [255, 0, 0])


class TestMetricsCalculator:
    """Test metrics calculation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "iou_thresholds": [0.5, 0.75],
            "confidence_threshold": 0.5
        }
        self.calculator = MetricsCalculator(self.config)
        
    def test_detection_diversity(self):
        """Test detection diversity calculation."""
        class_counts = {"building": 5, "vehicle": 3, "road": 2}
        diversity = self.calculator._calculate_detection_diversity(class_counts)
        assert diversity > 0
        
        # Single class should have zero diversity
        single_class = {"building": 10}
        diversity_single = self.calculator._calculate_detection_diversity(single_class)
        assert diversity_single == 0
        
    def test_edge_consistency(self):
        """Test edge consistency calculation."""
        # Create smooth segmentation
        predictions = torch.ones(10, 10)
        consistency = self.calculator._calculate_edge_consistency(predictions)
        assert consistency > 0
        
    def test_spatial_correlation(self):
        """Test spatial correlation calculation."""
        # Mock detection results
        detection_results = {
            "count": 2,
            "boxes": torch.tensor([[0, 0, 5, 5], [10, 10, 15, 15]])
        }
        
        # Mock segmentation results
        segmentation_results = {
            "predictions": torch.zeros(20, 20)
        }
        # Add some non-background pixels in detection areas
        segmentation_results["predictions"][0:5, 0:5] = 1
        segmentation_results["predictions"][10:15, 10:15] = 2
        
        correlation = self.calculator._calculate_spatial_correlation(
            detection_results, segmentation_results
        )
        assert correlation["correlation_score"] > 0


class TestVisualizer:
    """Test visualization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "dpi": 100,
            "figsize": (8, 6),
            "style": "satellite"
        }
        self.visualizer = Visualizer(self.config)
        
    def test_detection_drawing(self):
        """Test detection box drawing."""
        image = np.random.rand(100, 100, 3)
        detection = {
            "count": 1,
            "boxes": torch.tensor([[10, 10, 50, 50]]),
            "scores": torch.tensor([0.9]),
            "class_names": ["building"]
        }
        
        vis_image = self.visualizer._draw_detections(image, detection)
        assert vis_image.shape == image.shape
        
    def test_timestamp_generation(self):
        """Test timestamp generation."""
        timestamp = self.visualizer._get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0


class TestGISAnalyzer:
    """Test main pipeline functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "preprocessing": {"target_size": [256, 256]},
            "detection": {"model_size": "n", "confidence_threshold": 0.5},
            "segmentation": {"num_classes": 3, "encoder_name": "resnet18"},
            "evaluation": {"iou_thresholds": [0.5]},
            "visualization": {"dpi": 100}
        }
        
    def test_component_initialization(self):
        """Test that all components initialize correctly."""
        # This test would require actual model files, so we'll mock it
        analyzer = GISAnalyzer.__new__(GISAnalyzer)
        analyzer.config = self.config
        analyzer.device = torch.device("cpu")
        analyzer.models_dir = Path("models")
        analyzer.batch_size = 1
        
        # Test configuration parsing
        assert analyzer.config["preprocessing"]["target_size"] == [256, 256]
        assert analyzer.config["detection"]["confidence_threshold"] == 0.5


def test_config_loading():
    """Test configuration loading."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
        test_param: 123
        nested:
          value: "test"
        """)
        temp_path = f.name
        
    try:
        config = load_config(temp_path)
        assert config["test_param"] == 123
        assert config["nested"]["value"] == "test"
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
