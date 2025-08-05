"""
Comprehensive evaluation metrics for object detection and segmentation
"""

import numpy as np
import torch
from typing import Dict, Any, List, Union, Tuple
import logging
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for detection and segmentation tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # IoU thresholds for evaluation
        self.iou_thresholds = config.get("iou_thresholds", [0.5, 0.75, 0.9])
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
    def calculate_metrics(
        self,
        detection_results: Dict[str, Any],
        segmentation_results: Dict[str, Any],
        image_data: Dict[str, Any],
        ground_truth: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for both detection and segmentation.
        
        Args:
            detection_results: Object detection results
            segmentation_results: Semantic segmentation results
            image_data: Original image data
            ground_truth: Optional ground truth annotations
            
        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}
        
        # Detection metrics
        metrics["detection"] = self._calculate_detection_metrics(detection_results, ground_truth)
        
        # Segmentation metrics
        metrics["segmentation"] = self._calculate_segmentation_metrics(segmentation_results, ground_truth)
        
        # Combined analysis
        metrics["combined"] = self._calculate_combined_metrics(
            detection_results, segmentation_results, image_data
        )
        
        # Image quality metrics
        metrics["image_quality"] = self._calculate_image_quality_metrics(image_data)
        
        self.logger.info("Metrics calculation completed")
        return metrics
        
    def _calculate_detection_metrics(
        self,
        detection_results: Dict[str, Any],
        ground_truth: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Calculate object detection metrics."""
        metrics = {}
        
        # Basic statistics
        metrics["total_detections"] = detection_results["count"]
        metrics["detection_summary"] = detection_results["detection_summary"]
        
        # Confidence distribution
        if len(detection_results["scores"]) > 0:
            scores = detection_results["scores"]
            metrics["confidence_stats"] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()) if len(scores) > 1 else 0.0,
                "min": float(scores.min()),
                "max": float(scores.max()),
                "median": float(scores.median())
            }
        else:
            metrics["confidence_stats"] = {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0
            }
            
        # Size distribution
        if len(detection_results["areas"]) > 0:
            areas = detection_results["areas"]
            metrics["size_distribution"] = {
                "mean_area": float(areas.mean()),
                "std_area": float(areas.std()) if len(areas) > 1 else 0.0,
                "min_area": float(areas.min()),
                "max_area": float(areas.max()),
                "small_objects": (areas < 1000).sum().item(),
                "medium_objects": ((areas >= 1000) & (areas < 10000)).sum().item(),
                "large_objects": (areas >= 10000).sum().item()
            }
        else:
            metrics["size_distribution"] = {
                "mean_area": 0.0, "std_area": 0.0, "min_area": 0.0, "max_area": 0.0,
                "small_objects": 0, "medium_objects": 0, "large_objects": 0
            }
            
        # Class distribution
        class_names = detection_results["class_names"]
        if class_names:
            from collections import Counter
            class_counts = Counter(class_names)
            metrics["class_distribution"] = dict(class_counts)
            metrics["diversity_index"] = self._calculate_detection_diversity(class_counts)
        else:
            metrics["class_distribution"] = {}
            metrics["diversity_index"] = 0.0
            
        # If ground truth is available, calculate precision/recall
        if ground_truth and "detection" in ground_truth:
            metrics.update(self._calculate_detection_precision_recall(detection_results, ground_truth["detection"]))
            
        return metrics
        
    def _calculate_segmentation_metrics(
        self,
        segmentation_results: Dict[str, Any],
        ground_truth: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Calculate semantic segmentation metrics."""
        metrics = {}
        
        # Basic statistics
        metrics["segmentation_summary"] = segmentation_results["segmentation_summary"]
        metrics["class_statistics"] = segmentation_results["class_statistics"]
        
        # Calculate pixel accuracy
        predictions = segmentation_results["predictions"]
        probabilities = segmentation_results["probabilities"]
        
        # Overall confidence
        max_probs = torch.max(probabilities, dim=0)[0]
        metrics["pixel_confidence"] = {
            "mean": float(max_probs.mean()),
            "std": float(max_probs.std()),
            "min": float(max_probs.min()),
            "max": float(max_probs.max())
        }
        
        # Edge consistency (measure of segmentation smoothness)
        metrics["edge_consistency"] = self._calculate_edge_consistency(predictions)
        
        # Region coherence
        metrics["region_coherence"] = self._calculate_region_coherence(predictions)
        
        # If ground truth is available, calculate IoU and other metrics
        if ground_truth and "segmentation" in ground_truth:
            gt_mask = ground_truth["segmentation"]
            metrics["iou_scores"] = self._calculate_segmentation_iou(predictions, gt_mask)
            metrics["pixel_accuracy"] = self._calculate_pixel_accuracy(predictions, gt_mask)
            metrics["class_accuracy"] = self._calculate_class_accuracy(predictions, gt_mask)
            
        return metrics
        
    def _calculate_combined_metrics(
        self,
        detection_results: Dict[str, Any],
        segmentation_results: Dict[str, Any],
        image_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics that combine detection and segmentation results."""
        metrics = {}
        
        # Spatial correlation between detections and segmentation
        metrics["spatial_correlation"] = self._calculate_spatial_correlation(
            detection_results, segmentation_results
        )
        
        # Coverage analysis
        metrics["coverage_analysis"] = self._calculate_coverage_analysis(
            detection_results, segmentation_results, image_data
        )
        
        # Consistency score
        metrics["consistency_score"] = self._calculate_consistency_score(
            detection_results, segmentation_results
        )
        
        return metrics
        
    def _calculate_image_quality_metrics(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate image quality metrics."""
        metrics = {}
        
        # Get RGB image
        rgb_image = image_data["normalized"]
        
        # Sharpness (using Laplacian variance)
        gray = np.mean(rgb_image, axis=2)
        laplacian = np.abs(np.gradient(gray, axis=0)) + np.abs(np.gradient(gray, axis=1))
        metrics["sharpness"] = float(np.var(laplacian))
        
        # Contrast (RMS contrast)
        metrics["contrast"] = float(np.std(gray))
        
        # Brightness
        metrics["brightness"] = float(np.mean(rgb_image))
        
        # Color distribution
        metrics["color_distribution"] = {
            "red_mean": float(np.mean(rgb_image[:, :, 0])),
            "green_mean": float(np.mean(rgb_image[:, :, 1])),
            "blue_mean": float(np.mean(rgb_image[:, :, 2])),
            "red_std": float(np.std(rgb_image[:, :, 0])),
            "green_std": float(np.std(rgb_image[:, :, 1])),
            "blue_std": float(np.std(rgb_image[:, :, 2]))
        }
        
        # Information content (entropy)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        metrics["entropy"] = float(entropy)
        
        return metrics
        
    def _calculate_detection_diversity(self, class_counts: Dict[str, int]) -> float:
        """Calculate detection diversity using Shannon entropy."""
        total = sum(class_counts.values())
        if total == 0:
            return 0.0
            
        entropy = 0.0
        for count in class_counts.values():
            p = count / total
            entropy -= p * np.log(p + 1e-10)
            
        return entropy
        
    def _calculate_edge_consistency(self, predictions: torch.Tensor) -> float:
        """Calculate edge consistency of segmentation."""
        # Convert to numpy
        pred_np = predictions.cpu().numpy().astype(np.float32)
        
        # Calculate gradients
        grad_x = np.abs(np.gradient(pred_np, axis=1))
        grad_y = np.abs(np.gradient(pred_np, axis=0))
        
        # Edge strength
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return inverse of edge variation (higher = more consistent)
        return float(1.0 / (np.std(edge_strength) + 1e-6))
        
    def _calculate_region_coherence(self, predictions: torch.Tensor) -> float:
        """Calculate region coherence of segmentation."""
        from scipy import ndimage
        
        pred_np = predictions.cpu().numpy()
        
        # Calculate region coherence for each class
        coherence_scores = []
        unique_classes = np.unique(pred_np)
        
        for class_id in unique_classes:
            class_mask = (pred_np == class_id).astype(np.uint8)
            if np.sum(class_mask) > 0:
                # Label connected components
                labeled, num_features = ndimage.label(class_mask)
                
                # Calculate coherence as inverse of fragmentation
                if num_features > 0:
                    total_area = np.sum(class_mask)
                    avg_region_size = total_area / num_features
                    coherence = avg_region_size / (total_area + 1e-6)
                    coherence_scores.append(coherence)
                    
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
    def _calculate_spatial_correlation(
        self,
        detection_results: Dict[str, Any],
        segmentation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate spatial correlation between detections and segmentation."""
        if detection_results["count"] == 0:
            return {"correlation_score": 0.0, "matched_detections": 0}
            
        boxes = detection_results["boxes"]
        predictions = segmentation_results["predictions"]
        
        matched_detections = 0
        
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            
            # Extract region from segmentation
            region = predictions[y1:y2, x1:x2]
            if region.numel() > 0:
                # Check if region is dominated by non-background class
                unique_classes, counts = torch.unique(region, return_counts=True)
                if len(unique_classes) > 1:  # More than just background
                    dominant_class = unique_classes[torch.argmax(counts)]
                    if dominant_class != 0:  # Not background
                        matched_detections += 1
                        
        correlation_score = matched_detections / detection_results["count"]
        
        return {
            "correlation_score": correlation_score,
            "matched_detections": matched_detections,
            "total_detections": detection_results["count"]
        }
        
    def _calculate_coverage_analysis(
        self,
        detection_results: Dict[str, Any],
        segmentation_results: Dict[str, Any],
        image_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate coverage analysis."""
        image_area = image_data["metadata"]["width"] * image_data["metadata"]["height"]
        
        # Detection coverage
        if detection_results["count"] > 0:
            detection_area = torch.sum(detection_results["areas"]).item()
            detection_coverage = detection_area / image_area
        else:
            detection_coverage = 0.0
            
        # Segmentation coverage (non-background pixels)
        predictions = segmentation_results["predictions"]
        non_background_pixels = (predictions != 0).sum().item()
        segmentation_coverage = non_background_pixels / predictions.numel()
        
        return {
            "detection_coverage": detection_coverage,
            "segmentation_coverage": segmentation_coverage,
            "coverage_ratio": detection_coverage / (segmentation_coverage + 1e-6)
        }
        
    def _calculate_consistency_score(
        self,
        detection_results: Dict[str, Any],
        segmentation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall consistency between detection and segmentation."""
        # This is a simplified consistency score
        # In practice, you might want more sophisticated metrics
        
        spatial_corr = self._calculate_spatial_correlation(detection_results, segmentation_results)
        coverage = self._calculate_coverage_analysis(detection_results, segmentation_results, {"metadata": {"width": 1024, "height": 1024}})
        
        # Combine different factors
        consistency = (
            spatial_corr["correlation_score"] * 0.5 +
            min(coverage["coverage_ratio"], 1.0) * 0.3 +
            (1.0 - abs(coverage["detection_coverage"] - coverage["segmentation_coverage"])) * 0.2
        )
        
        return consistency
        
    def _calculate_segmentation_iou(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate IoU for segmentation classes."""
        num_classes = int(max(predictions.max(), ground_truth.max())) + 1
        iou_scores = {}
        
        for class_id in range(num_classes):
            pred_mask = (predictions == class_id)
            gt_mask = (ground_truth == class_id)
            
            intersection = (pred_mask & gt_mask).sum().float()
            union = (pred_mask | gt_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0
                
            iou_scores[f"class_{class_id}"] = iou.item()
            
        # Mean IoU
        iou_scores["mean_iou"] = np.mean(list(iou_scores.values()))
        
        return iou_scores
        
    def _calculate_pixel_accuracy(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """Calculate pixel accuracy."""
        correct_pixels = (predictions == ground_truth).sum().float()
        total_pixels = ground_truth.numel()
        
        return float(correct_pixels / total_pixels)
        
    def _calculate_class_accuracy(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate per-class accuracy."""
        num_classes = int(max(predictions.max(), ground_truth.max())) + 1
        class_accuracies = {}
        
        for class_id in range(num_classes):
            gt_mask = (ground_truth == class_id)
            if gt_mask.sum() > 0:
                pred_mask = (predictions == class_id)
                correct = (pred_mask & gt_mask).sum().float()
                total = gt_mask.sum().float()
                accuracy = correct / total
            else:
                accuracy = 0.0
                
            class_accuracies[f"class_{class_id}"] = accuracy.item()
            
        return class_accuracies
