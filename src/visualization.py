"""
Visualization utilities for GIS image analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import logging


class Visualizer:
    """Creates visualizations for GIS analysis results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Define color palette for land cover classes
        self.land_cover_colors = {
            0: '#FF6B6B',  # Urban - Red
            1: '#4ECDC4',  # Forest - Green
            2: '#45B7D1',  # Water - Blue
            3: '#96CEB4',  # Agriculture - Light Green
            4: '#FFEAA7',  # Bare Soil - Yellow
        }
        
        self.land_cover_labels = {
            0: 'Urban',
            1: 'Forest', 
            2: 'Water',
            3: 'Agriculture',
            4: 'Bare Soil'
        }
    
    def create_land_cover_map(self, processed_data: Dict, predictions: np.ndarray,
                             save_as: str = 'land_cover_map.png'):
        """
        Create land cover classification map.
        
        Args:
            processed_data: Processed image data
            predictions: Model predictions
            save_as: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        image = processed_data['image']
        if image.ndim == 3 and image.shape[-1] >= 3:
            # RGB visualization
            rgb_image = image[:, :, :3]
            # Normalize for display
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            axes[0].imshow(rgb_image)
        else:
            # Grayscale
            axes[0].imshow(image.squeeze(), cmap='gray')
        
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Land cover map
        # Create custom colormap
        unique_classes = np.unique(predictions)
        colors = [self.land_cover_colors.get(cls, '#808080') for cls in unique_classes]
        cmap = mcolors.ListedColormap(colors)
        
        im = axes[1].imshow(predictions, cmap=cmap, vmin=0, vmax=len(unique_classes)-1)
        axes[1].set_title('Land Cover Classification')
        axes[1].axis('off')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
        cbar.set_ticks(unique_classes)
        cbar.set_ticklabels([self.land_cover_labels.get(cls, f'Class {cls}') 
                            for cls in unique_classes])
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vis_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Land cover map saved: {output_path}")
    
    def create_overlay_visualization(self, original_image_path: str, 
                                   predictions: np.ndarray,
                                   alpha: float = 0.6,
                                   save_as: str = 'overlay_visualization.png'):
        """
        Create overlay visualization of predictions on original image.
        
        Args:
            original_image_path: Path to original TIFF image
            predictions: Model predictions
            alpha: Transparency of overlay
            save_as: Output filename
        """
        try:
            # Try to load original image
            import rasterio
            with rasterio.open(original_image_path) as src:
                original = src.read()
                if original.ndim == 3:
                    original = np.transpose(original, (1, 2, 0))
        except:
            # Fallback: create dummy image
            self.logger.warning("Could not load original image, creating dummy")
            original = np.random.rand(*predictions.shape, 3)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Original image
        if original.ndim == 3:
            # Normalize for display
            original_norm = (original - original.min()) / (original.max() - original.min())
            axes[0].imshow(original_norm[:, :, :3] if original.shape[-1] >= 3 else original_norm)
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predictions only
        unique_classes = np.unique(predictions)
        colors = [self.land_cover_colors.get(cls, '#808080') for cls in unique_classes]
        cmap = mcolors.ListedColormap(colors)
        
        im = axes[1].imshow(predictions, cmap=cmap, vmin=0, vmax=len(unique_classes)-1)
        axes[1].set_title('Land Cover Classification')
        axes[1].axis('off')
        
        # Overlay
        if original.ndim == 3:
            base_image = original_norm[:, :, :3] if original.shape[-1] >= 3 else np.stack([original_norm]*3, axis=-1)
        else:
            base_image = np.stack([original]*3, axis=-1)
        
        axes[2].imshow(base_image)
        overlay = axes[2].imshow(predictions, cmap=cmap, alpha=alpha, 
                               vmin=0, vmax=len(unique_classes)-1)
        axes[2].set_title('Overlay Visualization')
        axes[2].axis('off')
        
        # Add shared colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.6, aspect=20)
        cbar.set_ticks(unique_classes)
        cbar.set_ticklabels([self.land_cover_labels.get(cls, f'Class {cls}') 
                            for cls in unique_classes])
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vis_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Overlay visualization saved: {output_path}")
    
    def create_feature_visualization(self, features: Dict, 
                                   save_as: str = 'feature_visualization.png'):
        """
        Visualize extracted features.
        
        Args:
            features: Dictionary containing extracted features
            save_as: Output filename
        """
        spectral_features = features.get('spectral_features')
        texture_features = features.get('texture_features')
        
        if spectral_features is None or texture_features is None:
            self.logger.warning("Missing features for visualization")
            return
        
        # Create subplot grid
        n_spectral = min(spectral_features.shape[-1], 4)
        n_texture = min(texture_features.shape[-1], 4)
        
        fig, axes = plt.subplots(2, max(n_spectral, n_texture), figsize=(16, 8))
        
        # Spectral features
        for i in range(n_spectral):
            if n_spectral == 1:
                ax = axes[0]
            else:
                ax = axes[0, i] if axes.ndim > 1 else axes[0]
            
            im = ax.imshow(spectral_features[:, :, i], cmap='viridis')
            ax.set_title(f'Spectral Feature {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Fill remaining spectral subplots
        for i in range(n_spectral, max(n_spectral, n_texture)):
            if axes.ndim > 1 and i < axes.shape[1]:
                axes[0, i].axis('off')
        
        # Texture features
        for i in range(n_texture):
            if n_texture == 1:
                ax = axes[1]
            else:
                ax = axes[1, i] if axes.ndim > 1 else axes[1]
            
            im = ax.imshow(texture_features[:, :, i], cmap='plasma')
            ax.set_title(f'Texture Feature {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Fill remaining texture subplots
        for i in range(n_texture, max(n_spectral, n_texture)):
            if axes.ndim > 1 and i < axes.shape[1]:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vis_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature visualization saved: {output_path}")
    
    def save_metrics_plot(self, metrics: Dict, save_as: str = 'metrics_plot.png'):
        """
        Create visualization of evaluation metrics.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            save_as: Output filename
        """
        # Create subplots for different metric types
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Class distribution pie chart
        class_counts = {}
        class_percentages = {}
        
        for key, value in metrics.items():
            if key.endswith('_count'):
                class_name = key.replace('_count', '').replace('class_', '')
                class_counts[class_name] = value
            elif key.endswith('_percentage'):
                class_name = key.replace('_percentage', '').replace('class_', '')
                class_percentages[class_name] = value
        
        if class_percentages:
            labels = [self.land_cover_labels.get(int(cls), f'Class {cls}') 
                     for cls in class_percentages.keys()]
            sizes = list(class_percentages.values())
            colors = [self.land_cover_colors.get(int(cls), '#808080') 
                     for cls in class_percentages.keys()]
            
            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0, 0].set_title('Class Distribution')
        
        # 2. IoU scores bar chart
        iou_scores = {}
        for key, value in metrics.items():
            if key.endswith('_iou') and key != 'mean_iou':
                class_name = key.replace('_iou', '').replace('class_', '')
                iou_scores[class_name] = value
        
        if iou_scores:
            classes = [self.land_cover_labels.get(int(cls), f'Class {cls}') 
                      for cls in iou_scores.keys()]
            scores = list(iou_scores.values())
            colors = [self.land_cover_colors.get(int(cls), '#808080') 
                     for cls in iou_scores.keys()]
            
            bars = axes[0, 1].bar(classes, scores, color=colors)
            axes[0, 1].set_title('IoU Scores by Class')
            axes[0, 1].set_ylabel('IoU Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Confusion matrix heatmap
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            
            # Create labels
            n_classes = cm.shape[0]
            class_labels = [self.land_cover_labels.get(i, f'Class {i}') 
                           for i in range(n_classes)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels,
                       ax=axes[1, 0])
            axes[1, 0].set_title('Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # 4. Summary metrics
        summary_metrics = {}
        for key in ['accuracy', 'mean_iou', 'mean_f1', 'mean_precision', 'mean_recall']:
            if key in metrics:
                summary_metrics[key.replace('mean_', '').replace('_', ' ').title()] = metrics[key]
        
        if summary_metrics:
            metric_names = list(summary_metrics.keys())
            metric_values = list(summary_metrics.values())
            
            bars = axes[1, 1].bar(metric_names, metric_values, 
                                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(metric_names)])
            axes[1, 1].set_title('Summary Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vis_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Metrics plot saved: {output_path}")
    
    def save_as_tiff(self, data: np.ndarray, metadata: Dict, 
                     filename: str = 'output.tiff'):
        """
        Save results as GeoTIFF with proper georeferencing.
        
        Args:
            data: Data array to save
            metadata: Metadata from original image
            filename: Output filename
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            output_path = os.path.join(self.vis_dir, filename)
            
            # Create profile
            profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype,
                'crs': metadata.get('crs'),
                'transform': metadata.get('transform'),
                'compress': 'lzw'
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            self.logger.info(f"GeoTIFF saved: {output_path}")
            
        except ImportError:
            self.logger.warning("rasterio not available, cannot save as GeoTIFF")
        except Exception as e:
            self.logger.error(f"Error saving GeoTIFF: {str(e)}")
    
    def create_comparison_grid(self, images: List[np.ndarray], 
                              titles: List[str],
                              save_as: str = 'comparison_grid.png',
                              cols: int = 3):
        """
        Create a grid comparison of multiple images.
        
        Args:
            images: List of image arrays
            titles: List of titles for each image
            save_as: Output filename
            cols: Number of columns in grid
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (image, title) in enumerate(zip(images, titles)):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            if image.ndim == 3:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap='viridis')
            
            ax.set_title(title)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.vis_dir, save_as)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison grid saved: {output_path}")
