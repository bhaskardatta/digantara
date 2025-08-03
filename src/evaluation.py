"""
Model evaluation utilities for GIS image analysis.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import json
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import jaccard_score


class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_accuracy(self, predictions: np.ndarray, 
                          ground_truth: Optional[np.ndarray] = None) -> float:
        """
        Calculate pixel-wise accuracy.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels (if available)
            
        Returns:
            Accuracy score
        """
        if ground_truth is None:
            self.logger.warning("No ground truth available for accuracy calculation")
            return 0.0
        
        # Flatten arrays for comparison
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten()
        
        accuracy = accuracy_score(gt_flat, pred_flat)
        return accuracy
    
    def calculate_iou(self, predictions: np.ndarray, 
                     ground_truth: Optional[np.ndarray] = None,
                     num_classes: int = 5) -> Dict[str, float]:
        """
        Calculate Intersection over Union (IoU) for each class.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with IoU scores per class and mean IoU
        """
        if ground_truth is None:
            self.logger.warning("No ground truth available for IoU calculation")
            return {'mean_iou': 0.0}
        
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten()
        
        iou_scores = {}
        ious = []
        
        for class_id in range(num_classes):
            # Calculate IoU for each class
            pred_mask = (pred_flat == class_id)
            gt_mask = (gt_flat == class_id)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union == 0:
                iou = 1.0  # Perfect score if class not present
            else:
                iou = intersection / union
            
            iou_scores[f'class_{class_id}_iou'] = iou
            ious.append(iou)
        
        iou_scores['mean_iou'] = np.mean(ious)
        return iou_scores
    
    def calculate_f1_scores(self, predictions: np.ndarray,
                           ground_truth: Optional[np.ndarray] = None,
                           num_classes: int = 5) -> Dict[str, float]:
        """
        Calculate F1 scores for each class.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with F1 scores
        """
        if ground_truth is None:
            self.logger.warning("No ground truth available for F1 calculation")
            return {'mean_f1': 0.0}
        
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten()
        
        # Calculate classification report
        report = classification_report(
            gt_flat, pred_flat, 
            labels=list(range(num_classes)),
            output_dict=True,
            zero_division=0
        )
        
        f1_scores = {}
        for class_id in range(num_classes):
            if str(class_id) in report:
                f1_scores[f'class_{class_id}_f1'] = report[str(class_id)]['f1-score']
            else:
                f1_scores[f'class_{class_id}_f1'] = 0.0
        
        f1_scores['mean_f1'] = report.get('macro avg', {}).get('f1-score', 0.0)
        
        return f1_scores
    
    def calculate_confusion_matrix(self, predictions: np.ndarray,
                                  ground_truth: Optional[np.ndarray] = None,
                                  num_classes: int = 5) -> Dict:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        if ground_truth is None:
            self.logger.warning("No ground truth available for confusion matrix")
            return {'confusion_matrix': np.zeros((num_classes, num_classes))}
        
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten()
        
        cm = confusion_matrix(gt_flat, pred_flat, labels=list(range(num_classes)))
        
        # Calculate per-class precision and recall
        precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-8)
        recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)
        
        return {
            'confusion_matrix': cm.tolist(),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'mean_precision': np.mean(precision),
            'mean_recall': np.mean(recall)
        }
    
    def calculate_class_distribution(self, predictions: np.ndarray) -> Dict:
        """
        Calculate class distribution in predictions.
        
        Args:
            predictions: Predicted labels
            
        Returns:
            Dictionary with class distribution statistics
        """
        pred_flat = predictions.flatten()
        unique, counts = np.unique(pred_flat, return_counts=True)
        total_pixels = len(pred_flat)
        
        distribution = {}
        for class_id, count in zip(unique, counts):
            distribution[f'class_{int(class_id)}_count'] = int(count)
            distribution[f'class_{int(class_id)}_percentage'] = float(count / total_pixels * 100)
        
        return distribution
    
    def calculate_spatial_metrics(self, predictions: np.ndarray) -> Dict:
        """
        Calculate spatial coherence metrics.
        
        Args:
            predictions: Predicted labels
            
        Returns:
            Dictionary with spatial metrics
        """
        # Calculate edge density as a measure of fragmentation
        h, w = predictions.shape
        edge_count = 0
        
        # Count horizontal edges
        for i in range(h):
            for j in range(w - 1):
                if predictions[i, j] != predictions[i, j + 1]:
                    edge_count += 1
        
        # Count vertical edges
        for i in range(h - 1):
            for j in range(w):
                if predictions[i, j] != predictions[i + 1, j]:
                    edge_count += 1
        
        total_possible_edges = (h * (w - 1)) + ((h - 1) * w)
        edge_density = edge_count / total_possible_edges if total_possible_edges > 0 else 0
        
        # Calculate largest connected component for each class
        class_components = {}
        unique_classes = np.unique(predictions)
        
        for class_id in unique_classes:
            mask = (predictions == class_id).astype(np.uint8)
            # Simple connected component analysis (8-connectivity)
            component_sizes = self._find_connected_components(mask)
            if component_sizes:
                class_components[f'class_{int(class_id)}_largest_component'] = max(component_sizes)
                class_components[f'class_{int(class_id)}_num_components'] = len(component_sizes)
            else:
                class_components[f'class_{int(class_id)}_largest_component'] = 0
                class_components[f'class_{int(class_id)}_num_components'] = 0
        
        return {
            'edge_density': edge_density,
            'fragmentation_index': edge_density,  # Higher = more fragmented
            **class_components
        }
    
    def _find_connected_components(self, binary_mask: np.ndarray) -> List[int]:
        """
        Find connected components in binary mask.
        
        Args:
            binary_mask: Binary mask
            
        Returns:
            List of component sizes
        """
        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []
        
        def dfs(i, j):
            if (i < 0 or i >= h or j < 0 or j >= w or 
                visited[i, j] or binary_mask[i, j] == 0):
                return 0
            
            visited[i, j] = True
            size = 1
            
            # 8-connectivity
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                          (0, 1), (1, -1), (1, 0), (1, 1)]:
                size += dfs(i + di, j + dj)
            
            return size
        
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] == 1 and not visited[i, j]:
                    component_size = dfs(i, j)
                    if component_size > 0:
                        components.append(component_size)
        
        return components
    
    def evaluate(self, predictions: np.ndarray, 
                ground_truth: Optional[np.ndarray] = None,
                num_classes: int = 5) -> Dict:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            predictions: Predicted labels
            ground_truth: Ground truth labels (optional)
            num_classes: Number of classes
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        metrics = {}
        
        # Basic accuracy metrics
        if ground_truth is not None:
            metrics['accuracy'] = self.calculate_accuracy(predictions, ground_truth)
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # IoU metrics
        iou_metrics = self.calculate_iou(predictions, ground_truth, num_classes)
        metrics.update(iou_metrics)
        
        # F1 scores
        f1_metrics = self.calculate_f1_scores(predictions, ground_truth, num_classes)
        metrics.update(f1_metrics)
        
        # Confusion matrix
        cm_metrics = self.calculate_confusion_matrix(predictions, ground_truth, num_classes)
        metrics.update(cm_metrics)
        
        # Class distribution
        distribution = self.calculate_class_distribution(predictions)
        metrics.update(distribution)
        
        # Spatial metrics
        spatial_metrics = self.calculate_spatial_metrics(predictions)
        metrics.update(spatial_metrics)
        
        # Add metadata
        metrics['evaluation_metadata'] = {
            'num_classes': num_classes,
            'prediction_shape': predictions.shape,
            'has_ground_truth': ground_truth is not None,
            'total_pixels': predictions.size
        }
        
        self.logger.info("Evaluation completed successfully")
        return metrics
    
    def save_results(self, metrics: Dict, filepath: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            metrics: Evaluation metrics dictionary
            filepath: Output file path
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            self.logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def print_summary(self, metrics: Dict):
        """
        Print evaluation summary to console.
        
        Args:
            metrics: Evaluation metrics dictionary
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'accuracy' in metrics:
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        
        if 'mean_iou' in metrics:
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
            
        if 'mean_f1' in metrics:
            print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
        
        print(f"\nClass Distribution:")
        for key, value in metrics.items():
            if key.endswith('_percentage'):
                class_name = key.replace('_percentage', '')
                print(f"  {class_name}: {value:.2f}%")
        
        if 'fragmentation_index' in metrics:
            print(f"\nSpatial Metrics:")
            print(f"  Fragmentation Index: {metrics['fragmentation_index']:.4f}")
        
        print("="*50)
