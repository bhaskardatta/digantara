"""
Advanced visualization and reporting tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import logging
import base64
from io import BytesIO


class Visualizer:
    """
    Advanced visualization and reporting for GIS analysis results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Visualization settings
        self.dpi = config.get("dpi", 300)
        self.figsize = config.get("figsize", (12, 8))
        self.style = config.get("style", "satellite")
        self.colormap = config.get("colormap", "viridis")
        
        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette("husl")
        
    def create_visualizations(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path]
    ) -> None:
        """
        Create comprehensive visualizations for analysis results.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory for visualizations
        """
        output_path = Path(output_dir)
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating visualizations...")
        
        # 1. Detection visualization
        self._create_detection_visualization(results, vis_dir)
        
        # 2. Segmentation visualization
        self._create_segmentation_visualization(results, vis_dir)
        
        # 3. Combined overlay
        self._create_combined_overlay(results, vis_dir)
        
        # 4. Metrics dashboard
        self._create_metrics_dashboard(results, vis_dir)
        
        # 5. Interactive map
        self._create_interactive_map(results, vis_dir)
        
        # 6. Detailed analysis plots
        self._create_analysis_plots(results, vis_dir)
        
        # 7. Generate HTML report
        self._generate_html_report(results, output_path)
        
        self.logger.info(f"Visualizations saved to {vis_dir}")
        
    def _create_detection_visualization(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create object detection visualization."""
        try:
            # Get image and detection results
            image = results["processed_image"]["normalized"]
            detection = results["detection"]
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Detection results
            vis_image = self._draw_detections(image.copy(), detection)
            axes[1].imshow(vis_image)
            axes[1].set_title(f"Detections (Count: {detection['count']})", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / "detection_results.png", dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Detection statistics plot
            self._plot_detection_statistics(detection, output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create detection visualization: {e}")
            
    def _create_segmentation_visualization(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create semantic segmentation visualization."""
        try:
            # Get image and segmentation results
            image = results["processed_image"]["normalized"]
            segmentation = results["segmentation"]
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Segmentation mask
            colored_mask = segmentation["colored_mask"]
            axes[0, 1].imshow(colored_mask)
            axes[0, 1].set_title("Segmentation Map", fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Overlay
            overlay = cv2.addWeighted(
                (image * 255).astype(np.uint8),
                0.6,
                colored_mask,
                0.4,
                0
            )
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title("Overlay", fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Class distribution
            class_stats = segmentation["class_statistics"]
            classes = list(class_stats.keys())
            percentages = [stats["percentage"] for stats in class_stats.values()]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            axes[1, 1].pie(percentages, labels=classes, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title("Land Use Distribution", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / "segmentation_results.png", dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Segmentation statistics
            self._plot_segmentation_statistics(segmentation, output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create segmentation visualization: {e}")
            
    def _create_combined_overlay(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create combined detection and segmentation overlay."""
        try:
            image = results["processed_image"]["normalized"]
            detection = results["detection"]
            segmentation = results["segmentation"]
            
            # Start with segmentation overlay
            colored_mask = segmentation["colored_mask"]
            overlay = cv2.addWeighted(
                (image * 255).astype(np.uint8),
                0.7,
                colored_mask,
                0.3,
                0
            )
            
            # Add detection boxes
            final_overlay = self._draw_detections(overlay / 255.0, detection)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(final_overlay)
            ax.set_title("Combined Analysis: Detection + Segmentation", fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add legend for segmentation
            legend_elements = []
            for i, class_name in enumerate(segmentation["class_names"]):
                if i < len(segmentation["class_names"]):
                    color = np.array(segmentation["colored_mask"]).max(axis=(0, 1))
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=np.array([0.2, 0.7, 0.3]), label=class_name)
                    )
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            plt.savefig(output_dir / "combined_overlay.png", dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create combined overlay: {e}")
            
    def _create_metrics_dashboard(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create comprehensive metrics dashboard."""
        try:
            metrics = results["metrics"]
            
            # Create dashboard with multiple subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    "Detection Confidence Distribution",
                    "Segmentation Class Distribution",
                    "Size Distribution",
                    "Quality Metrics",
                    "Spatial Correlation",
                    "Overall Performance"
                ],
                specs=[
                    [{"type": "histogram"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "indicator"}]
                ]
            )
            
            # Detection confidence distribution
            if "detection" in metrics and "confidence_stats" in metrics["detection"]:
                conf_stats = metrics["detection"]["confidence_stats"]
                fig.add_trace(
                    go.Histogram(
                        x=[conf_stats["mean"]] * 10,  # Placeholder
                        name="Confidence",
                        marker_color="blue"
                    ),
                    row=1, col=1
                )
            
            # Segmentation class distribution
            if "segmentation" in metrics and "class_statistics" in metrics["segmentation"]:
                class_stats = metrics["segmentation"]["class_statistics"]
                classes = list(class_stats.keys())
                percentages = [stats["percentage"] for stats in class_stats.values()]
                
                fig.add_trace(
                    go.Bar(
                        x=classes,
                        y=percentages,
                        name="Land Use %",
                        marker_color="green"
                    ),
                    row=1, col=2
                )
            
            # Size distribution
            if "detection" in metrics and "size_distribution" in metrics["detection"]:
                size_dist = metrics["detection"]["size_distribution"]
                sizes = ["Small", "Medium", "Large"]
                counts = [
                    size_dist["small_objects"],
                    size_dist["medium_objects"],
                    size_dist["large_objects"]
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=sizes,
                        y=counts,
                        name="Object Sizes",
                        marker_color="orange"
                    ),
                    row=2, col=1
                )
            
            # Quality metrics
            if "image_quality" in metrics:
                quality = metrics["image_quality"]
                fig.add_trace(
                    go.Scatter(
                        x=["Sharpness", "Contrast", "Brightness"],
                        y=[quality.get("sharpness", 0), quality.get("contrast", 0), quality.get("brightness", 0)],
                        mode="markers+lines",
                        name="Quality",
                        marker_color="red"
                    ),
                    row=2, col=2
                )
            
            # Spatial correlation
            if "combined" in metrics and "spatial_correlation" in metrics["combined"]:
                corr = metrics["combined"]["spatial_correlation"]
                fig.add_trace(
                    go.Bar(
                        x=["Matched", "Total"],
                        y=[corr["matched_detections"], corr["total_detections"]],
                        name="Correlation",
                        marker_color="purple"
                    ),
                    row=3, col=1
                )
            
            # Overall performance indicator
            if "combined" in metrics and "consistency_score" in metrics["combined"]:
                consistency = metrics["combined"]["consistency_score"]
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=consistency * 100,
                        title={"text": "Consistency Score"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "yellow"},
                                {"range": [80, 100], "color": "green"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90
                            }
                        }
                    ),
                    row=3, col=2
                )
            
            fig.update_layout(
                title_text="GIS Analysis Metrics Dashboard",
                title_x=0.5,
                height=1000,
                showlegend=False
            )
            
            fig.write_html(str(output_dir / "metrics_dashboard.html"))
            fig.write_image(str(output_dir / "metrics_dashboard.png"), width=1200, height=1000)
            
        except Exception as e:
            self.logger.error(f"Failed to create metrics dashboard: {e}")
            
    def _create_interactive_map(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create interactive map visualization."""
        try:
            # Create base map (using dummy coordinates for demo)
            # In real implementation, use actual geospatial coordinates
            m = folium.Map(
                location=[40.7128, -74.0060],  # Default to NYC
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add detection markers
            detection = results["detection"]
            if detection["count"] > 0:
                boxes = detection["boxes"]
                class_names = detection["class_names"]
                scores = detection["scores"]
                
                for i, (box, class_name, score) in enumerate(zip(boxes, class_names, scores)):
                    # Convert box to map coordinates (simplified)
                    lat = 40.7128 + (box[1].item() / 1000) * 0.001
                    lon = -74.0060 + (box[0].item() / 1000) * 0.001
                    
                    folium.Marker(
                        [lat, lon],
                        popup=f"{class_name}: {score:.2f}",
                        tooltip=f"Detection {i+1}",
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
            
            # Add segmentation overlay (simplified)
            # In real implementation, convert segmentation to GeoJSON
            
            # Save map
            m.save(str(output_dir / "interactive_map.html"))
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive map: {e}")
            
    def _create_analysis_plots(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Create detailed analysis plots."""
        try:
            # Detection analysis
            if results["detection"]["count"] > 0:
                self._plot_detection_analysis(results["detection"], output_dir)
            
            # Segmentation analysis
            self._plot_segmentation_analysis(results["segmentation"], output_dir)
            
            # Combined analysis
            self._plot_combined_analysis(results["metrics"]["combined"], output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis plots: {e}")
            
    def _draw_detections(
        self,
        image: np.ndarray,
        detection: Dict[str, Any]
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        if detection["count"] == 0:
            return image
            
        vis_image = image.copy()
        if vis_image.dtype == np.float32:
            vis_image = (vis_image * 255).astype(np.uint8)
        else:
            vis_image = vis_image.astype(np.uint8)
            
        boxes = detection["boxes"]
        scores = detection["scores"]
        class_names = detection["class_names"]
        
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for i, (box, score, class_name) in enumerate(zip(boxes, scores, class_names)):
            x1, y1, x2, y2 = box.int().tolist()
            color = colors[i % len(colors)]
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
        return vis_image.astype(np.float32) / 255.0
        
    def _plot_detection_statistics(
        self,
        detection: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Plot detailed detection statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        if detection["class_names"]:
            from collections import Counter
            class_counts = Counter(detection["class_names"])
            
            axes[0, 0].bar(class_counts.keys(), class_counts.values())
            axes[0, 0].set_title("Detection Class Distribution")
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        if len(detection["scores"]) > 0:
            axes[0, 1].hist(detection["scores"].cpu().numpy(), bins=20, alpha=0.7)
            axes[0, 1].set_title("Confidence Score Distribution")
            axes[0, 1].set_xlabel("Confidence")
            axes[0, 1].set_ylabel("Count")
        
        # Size distribution
        if len(detection["areas"]) > 0:
            axes[1, 0].hist(detection["areas"].cpu().numpy(), bins=20, alpha=0.7)
            axes[1, 0].set_title("Object Size Distribution")
            axes[1, 0].set_xlabel("Area (pixels)")
            axes[1, 0].set_ylabel("Count")
        
        # Summary statistics
        summary = detection["detection_summary"]
        stats_text = f"""
        Total Detections: {summary['total_detections']}
        Unique Classes: {summary['unique_classes']}
        Avg Confidence: {summary['average_confidence']:.3f}
        Max Confidence: {summary['max_confidence']:.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "detection_statistics.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _plot_segmentation_statistics(
        self,
        segmentation: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Plot detailed segmentation statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        class_stats = segmentation["class_statistics"]
        
        # Class percentages
        classes = list(class_stats.keys())
        percentages = [stats["percentage"] for stats in class_stats.values()]
        
        axes[0, 0].bar(classes, percentages)
        axes[0, 0].set_title("Land Use Class Distribution")
        axes[0, 0].set_ylabel("Percentage")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence by class
        confidences = [stats["average_confidence"] for stats in class_stats.values()]
        axes[0, 1].bar(classes, confidences)
        axes[0, 1].set_title("Average Confidence by Class")
        axes[0, 1].set_ylabel("Confidence")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Pixel counts
        pixel_counts = [stats["pixel_count"] for stats in class_stats.values()]
        axes[1, 0].bar(classes, pixel_counts)
        axes[1, 0].set_title("Pixel Count by Class")
        axes[1, 0].set_ylabel("Pixels")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary
        summary = segmentation["segmentation_summary"]
        summary_text = f"""
        Dominant Classes:
        """
        for class_name, percentage in summary["dominant_classes"]:
            summary_text += f"  {class_name}: {percentage:.1f}%\n"
        
        summary_text += f"""
        Overall Confidence: {summary['overall_confidence']:.3f}
        Classes Detected: {summary['total_classes_detected']}
        Diversity Index: {summary['land_use_diversity']:.3f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title("Segmentation Summary")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "segmentation_statistics.png", dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _plot_detection_analysis(self, detection: Dict[str, Any], output_dir: Path) -> None:
        """Create detailed detection analysis plots."""
        # Detection density heatmap, confidence vs size scatter, etc.
        pass
        
    def _plot_segmentation_analysis(self, segmentation: Dict[str, Any], output_dir: Path) -> None:
        """Create detailed segmentation analysis plots."""
        # Segmentation quality metrics, region coherence, etc.
        pass
        
    def _plot_combined_analysis(self, combined_metrics: Dict[str, Any], output_dir: Path) -> None:
        """Create combined analysis plots."""
        # Spatial correlation plots, coverage analysis, etc.
        pass
        
    def _generate_html_report(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Generate comprehensive HTML report."""
        try:
            # Read visualization images and encode as base64
            vis_dir = output_dir / "visualizations"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GIS Image Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
                    .image {{ text-align: center; margin: 20px 0; }}
                    .image img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>GIS Image Analysis Report</h1>
                    <p>Generated on {self._get_timestamp()}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    {self._create_executive_summary(results)}
                </div>
                
                <div class="section">
                    <h2>Detection Results</h2>
                    {self._create_detection_summary_html(results["detection"])}
                </div>
                
                <div class="section">
                    <h2>Segmentation Results</h2>
                    {self._create_segmentation_summary_html(results["segmentation"])}
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {self._create_visualizations_html(vis_dir)}
                </div>
                
                <div class="section">
                    <h2>Technical Metrics</h2>
                    {self._create_metrics_table_html(results["metrics"])}
                </div>
            </body>
            </html>
            """
            
            with open(output_dir / "analysis_report.html", "w") as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def _create_executive_summary(self, results: Dict[str, Any]) -> str:
        """Create executive summary HTML."""
        detection = results["detection"]
        segmentation = results["segmentation"]
        
        return f"""
        <p>This report presents the analysis of a GIS image using advanced AI/ML techniques.</p>
        <ul>
            <li><strong>Objects Detected:</strong> {detection['count']} objects across {detection['detection_summary']['unique_classes']} different classes</li>
            <li><strong>Land Use Classification:</strong> {segmentation['segmentation_summary']['total_classes_detected']} distinct land use types identified</li>
            <li><strong>Dominant Land Use:</strong> {segmentation['segmentation_summary']['dominant_classes'][0][0] if segmentation['segmentation_summary']['dominant_classes'] else 'None'}</li>
            <li><strong>Overall Confidence:</strong> {segmentation['segmentation_summary']['overall_confidence']:.2f}</li>
        </ul>
        """
        
    def _create_detection_summary_html(self, detection: Dict[str, Any]) -> str:
        """Create detection summary HTML."""
        if detection["count"] == 0:
            return "<p>No objects detected in the image.</p>"
            
        summary = detection["detection_summary"]
        
        html = f"""
        <h3>Detection Statistics</h3>
        <div class="metric">
            <strong>Total Detections:</strong> {summary['total_detections']}
        </div>
        <div class="metric">
            <strong>Unique Classes:</strong> {summary['unique_classes']}
        </div>
        <div class="metric">
            <strong>Average Confidence:</strong> {summary['average_confidence']:.3f}
        </div>
        
        <h4>Class Distribution</h4>
        <table>
            <tr><th>Class</th><th>Count</th></tr>
        """
        
        for class_name, count in summary["class_distribution"].items():
            html += f"<tr><td>{class_name}</td><td>{count}</td></tr>"
            
        html += "</table>"
        return html
        
    def _create_segmentation_summary_html(self, segmentation: Dict[str, Any]) -> str:
        """Create segmentation summary HTML."""
        class_stats = segmentation["class_statistics"]
        summary = segmentation["segmentation_summary"]
        
        html = f"""
        <h3>Land Use Analysis</h3>
        <div class="metric">
            <strong>Classes Detected:</strong> {summary['total_classes_detected']}
        </div>
        <div class="metric">
            <strong>Diversity Index:</strong> {summary['land_use_diversity']:.3f}
        </div>
        <div class="metric">
            <strong>Overall Confidence:</strong> {summary['overall_confidence']:.3f}
        </div>
        
        <h4>Land Use Distribution</h4>
        <table>
            <tr><th>Land Use Type</th><th>Percentage</th><th>Pixel Count</th><th>Confidence</th></tr>
        """
        
        for class_name, stats in class_stats.items():
            html += f"""
            <tr>
                <td>{class_name}</td>
                <td>{stats['percentage']:.1f}%</td>
                <td>{stats['pixel_count']:,}</td>
                <td>{stats['average_confidence']:.3f}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        
    def _create_visualizations_html(self, vis_dir: Path) -> str:
        """Create visualizations section HTML."""
        html = ""
        
        # List of visualization files to include
        viz_files = [
            ("detection_results.png", "Object Detection Results"),
            ("segmentation_results.png", "Semantic Segmentation Results"),
            ("combined_overlay.png", "Combined Analysis Overlay"),
            ("metrics_dashboard.png", "Metrics Dashboard")
        ]
        
        for filename, title in viz_files:
            file_path = vis_dir / filename
            if file_path.exists():
                html += f"""
                <div class="image">
                    <h4>{title}</h4>
                    <img src="visualizations/{filename}" alt="{title}">
                </div>
                """
                
        return html
        
    def _create_metrics_table_html(self, metrics: Dict[str, Any]) -> str:
        """Create metrics table HTML."""
        html = "<table><tr><th>Metric Category</th><th>Metric</th><th>Value</th></tr>"
        
        # Flatten metrics for table display
        def add_metrics_recursive(data: Dict[str, Any], category: str = ""):
            for key, value in data.items():
                if isinstance(value, dict):
                    add_metrics_recursive(value, f"{category}/{key}" if category else key)
                else:
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    html_row = f"<tr><td>{category}</td><td>{key}</td><td>{value_str}</td></tr>"
                    nonlocal html
                    html += html_row
                    
        add_metrics_recursive(metrics)
        html += "</table>"
        
        return html
