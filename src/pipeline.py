"""
Main GIS Analysis Pipeline
Orchestrates preprocessing, detection, segmentation, and visualization
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from .preprocessing.image_processor import ImageProcessor
from .detection.yolo_detector import YOLODetector
from .segmentation.deeplabv3_segmenter import DeepLabV3Segmenter
from .evaluation.metrics import MetricsCalculator
from .visualization.visualizer import Visualizer
from .utils.device import get_device


class GISAnalyzer:
    """
    Main pipeline for GIS image analysis combining object detection and semantic segmentation.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        models_dir: Union[str, Path] = "models/",
        device: str = "auto",
        batch_size: int = 1
    ):
        """
        Initialize GIS Analyzer.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory containing model weights
            device: Device to use for inference ('auto', 'cpu', 'cuda')
            batch_size: Batch size for processing
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.batch_size = batch_size
        self.device = get_device(device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing GIS analysis components...")
        
        # Image processor
        self.image_processor = ImageProcessor(
            config=self.config.get("preprocessing", {}),
            device=self.device
        )
        
        # Object detector
        self.detector = YOLODetector(
            config=self.config.get("detection", {}),
            models_dir=self.models_dir,
            device=self.device
        )
        
        # Semantic segmenter
        self.segmenter = DeepLabV3Segmenter(
            config=self.config.get("segmentation", {}),
            models_dir=self.models_dir,
            device=self.device
        )
        
        # Metrics calculator
        self.metrics = MetricsCalculator(
            config=self.config.get("evaluation", {})
        )
        
        # Visualizer
        self.visualizer = Visualizer(
            config=self.config.get("visualization", {})
        )
        
        self.logger.info("All components initialized successfully")
        
    def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a single TIFF image.
        
        Args:
            image_path: Path to TIFF image
            
        Returns:
            Analysis results dictionary
        """
        image_path = Path(image_path)
        self.logger.info(f"Analyzing image: {image_path}")
        
        import time
        start_time = time.time()
        
        try:
            # 1. Load and preprocess image
            self.logger.info("Step 1: Loading and preprocessing image...")
            image_data = self.image_processor.load_and_preprocess(image_path)
            load_time = time.time() - start_time
            self.logger.info(f"Image loaded and preprocessed in {load_time:.2f}s")
            
            # 2. Object detection
            self.logger.info("Step 2: Running object detection...")
            detection_start = time.time()
            detection_results = self.detector.detect(image_data["rgb"])
            detection_time = time.time() - detection_start
            self.logger.info(f"Detection completed in {detection_time:.2f}s")
            
            # 3. Semantic segmentation
            self.logger.info("Step 3: Running semantic segmentation...")
            segmentation_start = time.time()
            segmentation_results = self.segmenter.segment(image_data["rgb"])
            segmentation_time = time.time() - segmentation_start
            self.logger.info(f"Segmentation completed in {segmentation_time:.2f}s")
            
            # 4. Calculate metrics
            self.logger.info("Step 4: Calculating metrics...")
            metrics_start = time.time()
            metrics_results = self.metrics.calculate_metrics(
                detection_results,
                segmentation_results,
                image_data
            )
            metrics_time = time.time() - metrics_start
            
            total_time = time.time() - start_time
            
            # 5. Combine results
            results = {
                "image_path": str(image_path),
                "image_metadata": image_data["metadata"],
                "detections": detection_results.get("boxes", []),
                "segmentation": segmentation_results,
                "metrics": metrics_results,
                "processing_info": {
                    "total_time": total_time,
                    "load_time": load_time,
                    "detection_time": detection_time,
                    "segmentation_time": segmentation_time,
                    "metrics_time": metrics_time
                }
            }
            
            self.logger.info(f"Analysis completed for {image_path} in {total_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {image_path}: {e}")
            # Return partial results if possible
            return {
                "image_path": str(image_path),
                "error": str(e),
                "processing_info": {
                    "total_time": time.time() - start_time,
                    "status": "failed"
                }
            }
            
    def analyze_batch(self, image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of paths to TIFF images
            
        Returns:
            Batch analysis results
        """
        results = {}
        for image_path in image_paths:
            try:
                results[str(image_path)] = self.analyze_image(image_path)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results[str(image_path)] = {"error": str(e)}
                
        return results
        
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        visualize: bool = True
    ) -> None:
        """
        Save analysis results to disk.
        
        Args:
            results: Analysis results
            output_dir: Output directory
            visualize: Whether to generate visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detection and segmentation results
        self._save_predictions(results, output_path)
        
        # Save metrics
        self._save_metrics(results, output_path)
        
        # Generate visualizations
        if visualize:
            self.visualizer.create_visualizations(results, output_path)
            
        self.logger.info(f"Results saved to {output_path}")
        
    def _save_predictions(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save prediction results in various formats."""
        # Save as JSON
        import json
        
        # Prepare serializable results
        serializable_results = {
            "image_path": results["image_path"],
            "image_metadata": results["image_metadata"],
            "detection": {
                "boxes": results["detection"]["boxes"].tolist() if torch.is_tensor(results["detection"]["boxes"]) else results["detection"]["boxes"],
                "scores": results["detection"]["scores"].tolist() if torch.is_tensor(results["detection"]["scores"]) else results["detection"]["scores"],
                "labels": results["detection"]["labels"].tolist() if torch.is_tensor(results["detection"]["labels"]) else results["detection"]["labels"],
                "class_names": results["detection"].get("class_names", [])
            },
            "segmentation": {
                "predictions": results["segmentation"]["predictions"].tolist() if torch.is_tensor(results["segmentation"]["predictions"]) else results["segmentation"]["predictions"],
                "class_names": results["segmentation"].get("class_names", [])
            },
            "metrics": results["metrics"]
        }
        
        with open(output_path / "predictions.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
            
    def _save_metrics(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save metrics to CSV and JSON formats."""
        import json
        import pandas as pd
        
        metrics = results["metrics"]
        
        # Save as JSON
        with open(output_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Save as CSV
        if isinstance(metrics, dict):
            df = pd.DataFrame([metrics])
            df.to_csv(output_path / "metrics.csv", index=False)
