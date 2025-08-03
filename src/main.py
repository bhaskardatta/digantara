#!/usr/bin/env python3
"""
Main GIS Image Analysis Pipeline
Comprehensive satellite imagery analysis using SegFormer and YOLO-World
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Local imports
from preprocessing import ImagePreprocessor
from feature_extraction import SatelliteFeatureExtractor
from models import SegFormerSatellite, YOLOWorldSatellite, ModelManager
from evaluation import ModelEvaluator, PerformanceAnalyzer
from visualization import Visualizer, SatelliteVisualizer
from utils import setup_logging, load_config, create_output_structure


class GISAnalysisPipeline:
    """
    Comprehensive GIS image analysis pipeline integrating:
    - Advanced preprocessing
    - SegFormer semantic segmentation
    - YOLO-World object detection
    - Multi-modal feature extraction
    - Comprehensive evaluation and visualization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GIS analysis pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        
        # Setup logging
        setup_logging(
            log_level=self.config.get('log_level', 'INFO'),
            log_file=self.config.get('log_file')
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GIS Analysis Pipeline...")
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.results = {}
        self.processing_stats = {}
        
        self.logger.info("GIS Analysis Pipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Core processing components
        self.preprocessor = ImagePreprocessor(self.config.get('preprocessing', {}))
        self.feature_extractor = SatelliteFeatureExtractor(self.config.get('feature_extraction', {}))
        
        # Deep learning models
        self.segformer_model = SegFormerSatellite(self.config.get('segformer', {}))
        self.yolo_model = YOLOWorldSatellite(self.config.get('yolo', {}))
        self.model_manager = ModelManager()
        
        # Analysis and visualization
        self.evaluator = ModelEvaluator(self.config.get('evaluation', {}))
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Visualization components
        self.visualizer = Visualizer(self.config.get('visualization', {}))
        self.satellite_visualizer = SatelliteVisualizer(self.config.get('satellite_viz', {}))
    
    def run_comprehensive_analysis(self, 
                                 input_path: str,
                                 output_dir: str,
                                 analysis_type: str = 'full') -> Dict:
        """
        Run comprehensive GIS image analysis.
        
        Args:
            input_path: Path to input TIFF image
            output_dir: Output directory path
            analysis_type: Type of analysis ('segmentation', 'detection', 'full')
            
        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()
        self.logger.info(f"Starting comprehensive analysis of {input_path}")
        
        # Create output structure
        output_path = Path(output_dir)
        create_output_structure(output_path)
        
        try:
            # Phase 1: Preprocessing
            self.logger.info("Phase 1: Image Preprocessing")
            preprocessed_data = self.preprocessor.preprocess_image(input_path)
            
            # Phase 2: Feature Extraction
            self.logger.info("Phase 2: Feature Extraction")
            feature_data = self.feature_extractor.extract_features(preprocessed_data)
            
            # Phase 3: Deep Learning Analysis
            analysis_results = {}
            
            if analysis_type in ['segmentation', 'full']:
                self.logger.info("Phase 3a: SegFormer Semantic Segmentation")
                segmentation_results = self._run_segmentation_analysis(
                    preprocessed_data, output_path / 'segmentation'
                )
                analysis_results['segmentation'] = segmentation_results
            
            if analysis_type in ['detection', 'full']:
                self.logger.info("Phase 3b: YOLO-World Object Detection")
                detection_results = self._run_detection_analysis(
                    preprocessed_data, output_path / 'detection'
                )
                analysis_results['detection'] = detection_results
            
            # Phase 4: Combined Analysis (for full analysis)
            if analysis_type == 'full':
                self.logger.info("Phase 4: Combined Multi-Modal Analysis")
                combined_results = self._run_combined_analysis(
                    preprocessed_data, feature_data, analysis_results, 
                    output_path / 'combined'
                )
                analysis_results['combined'] = combined_results
            
            # Phase 5: Evaluation and Metrics
            self.logger.info("Phase 5: Performance Evaluation")
            evaluation_results = self._run_evaluation(
                preprocessed_data, analysis_results, output_path / 'evaluation'
            )
            
            # Phase 6: Visualization
            self.logger.info("Phase 6: Results Visualization")
            visualization_results = self._generate_visualizations(
                preprocessed_data, feature_data, analysis_results, 
                output_path / 'visualizations'
            )
            
            # Compile final results
            final_results = {
                'input_info': {
                    'input_path': input_path,
                    'output_dir': str(output_path),
                    'analysis_type': analysis_type,
                    'processing_time': time.time() - start_time
                },
                'preprocessing': preprocessed_data,
                'features': feature_data,
                'analysis': analysis_results,
                'evaluation': evaluation_results,
                'visualizations': visualization_results,
                'processing_stats': self.processing_stats
            }
            
            # Save results summary
            self._save_results_summary(final_results, output_path)
            
            self.logger.info(f"Analysis completed successfully in {time.time() - start_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
    
    def _run_segmentation_analysis(self, preprocessed_data: Dict, output_path: Path) -> Dict:
        """Run semantic segmentation analysis using SegFormer."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        image = preprocessed_data['image']
        
        # Run SegFormer inference
        segmentation_mask = self.segformer_model.segment_image(image)
        
        # Analyze segmentation results
        seg_analysis = self.segformer_model.analyze_segmentation(
            image, segmentation_mask
        )
        
        # Save segmentation outputs
        self.segformer_model.save_segmentation_results(
            image, segmentation_mask, seg_analysis, str(output_path)
        )
        
        results = {
            'segmentation_mask': segmentation_mask,
            'land_cover_analysis': seg_analysis,
            'model_info': self.segformer_model.get_model_info(),
            'output_files': list(output_path.glob('*'))
        }
        
        return results
    
    def _run_detection_analysis(self, preprocessed_data: Dict, output_path: Path) -> Dict:
        """Run object detection analysis using YOLO-World."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        image = preprocessed_data['image']
        
        # Run YOLO-World detection
        detections = self.yolo_model.detect_objects(image)
        
        # Analyze detection results
        detection_analysis = self.yolo_model.analyze_detections(detections, image.shape[:2])
        
        # Save detection outputs
        self.yolo_model.save_detection_results(
            image, detections, detection_analysis, str(output_path)
        )
        
        results = {
            'detections': detections,
            'detection_analysis': detection_analysis,
            'model_info': self.yolo_model.get_model_info(),
            'output_files': list(output_path.glob('*'))
        }
        
        return results
    
    def _run_combined_analysis(self, 
                             preprocessed_data: Dict,
                             feature_data: Dict,
                             analysis_results: Dict,
                             output_path: Path) -> Dict:
        """Run combined multi-modal analysis."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract results from individual analyses
        segmentation_results = analysis_results.get('segmentation', {})
        detection_results = analysis_results.get('detection', {})
        
        # Multi-modal fusion
        combined_analysis = self._fuse_multimodal_results(
            preprocessed_data, feature_data, segmentation_results, detection_results
        )
        
        # Generate combined visualizations
        combined_viz = self._create_combined_visualizations(
            preprocessed_data, combined_analysis, output_path
        )
        
        results = {
            'multimodal_fusion': combined_analysis,
            'combined_visualizations': combined_viz,
            'fusion_metrics': self._calculate_fusion_metrics(combined_analysis)
        }
        
        return results
    
    def _fuse_multimodal_results(self,
                               preprocessed_data: Dict,
                               feature_data: Dict,
                               segmentation_results: Dict,
                               detection_results: Dict) -> Dict:
        """Fuse results from multiple analysis modalities."""
        image_shape = preprocessed_data['image'].shape[:2]
        
        # Initialize fusion result
        fusion_result = {
            'land_cover_map': np.zeros(image_shape, dtype=np.int32),
            'confidence_map': np.zeros(image_shape, dtype=np.float32),
            'object_density_map': np.zeros(image_shape, dtype=np.float32),
            'integrated_features': {}
        }
        
        # Integrate segmentation results
        if 'segmentation_mask' in segmentation_results:
            seg_mask = segmentation_results['segmentation_mask']
            fusion_result['land_cover_map'] = seg_mask
            
            # Use segmentation confidence if available
            if 'confidence' in segmentation_results:
                fusion_result['confidence_map'] = segmentation_results['confidence']
        
        # Integrate detection results
        if 'detections' in detection_results:
            detections = detection_results['detections']
            
            # Create object density map
            for detection in detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
                    
                    confidence = detection.get('confidence', 0.5)
                    fusion_result['object_density_map'][y1:y2, x1:x2] += confidence
        
        # Integrate traditional features
        if 'comprehensive_features' in feature_data:
            comp_features = feature_data['comprehensive_features']
            
            # Select key features for integration
            key_features = ['vegetation_index', 'water_index', 'urban_index', 'brightness']
            for feature_name in key_features:
                if feature_name in comp_features:
                    fusion_result['integrated_features'][feature_name] = comp_features[feature_name]
        
        return fusion_result
    
    def _create_combined_visualizations(self,
                                      preprocessed_data: Dict,
                                      combined_analysis: Dict,
                                      output_path: Path) -> Dict:
        """Create combined visualizations."""
        viz_results = {}
        
        # Multi-modal overlay
        overlay_path = output_path / 'multimodal_overlay.png'
        self.satellite_visualizer.create_multimodal_overlay(
            preprocessed_data['image'],
            combined_analysis['land_cover_map'],
            combined_analysis['object_density_map'],
            str(overlay_path)
        )
        viz_results['multimodal_overlay'] = str(overlay_path)
        
        # Confidence visualization
        confidence_path = output_path / 'confidence_map.png'
        self.visualizer.plot_heatmap(
            combined_analysis['confidence_map'],
            'Analysis Confidence',
            str(confidence_path)
        )
        viz_results['confidence_map'] = str(confidence_path)
        
        return viz_results
    
    def _calculate_fusion_metrics(self, combined_analysis: Dict) -> Dict:
        """Calculate metrics for multimodal fusion."""
        metrics = {}
        
        # Land cover statistics
        land_cover_map = combined_analysis['land_cover_map']
        unique_classes, class_counts = np.unique(land_cover_map, return_counts=True)
        
        total_pixels = land_cover_map.size
        for cls, count in zip(unique_classes, class_counts):
            metrics[f'class_{cls}_percentage'] = (count / total_pixels) * 100
        
        # Confidence statistics
        confidence_map = combined_analysis['confidence_map']
        metrics['mean_confidence'] = float(np.mean(confidence_map))
        metrics['min_confidence'] = float(np.min(confidence_map))
        metrics['max_confidence'] = float(np.max(confidence_map))
        
        # Object density statistics
        obj_density = combined_analysis['object_density_map']
        metrics['mean_object_density'] = float(np.mean(obj_density))
        metrics['max_object_density'] = float(np.max(obj_density))
        
        return metrics
    
    def _run_evaluation(self, 
                       preprocessed_data: Dict,
                       analysis_results: Dict,
                       output_path: Path) -> Dict:
        """Run comprehensive evaluation."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        evaluation_results = {}
        
        # Evaluate segmentation if available
        if 'segmentation' in analysis_results:
            seg_eval = self.evaluator.evaluate_segmentation(
                analysis_results['segmentation']
            )
            evaluation_results['segmentation_metrics'] = seg_eval
        
        # Evaluate detection if available
        if 'detection' in analysis_results:
            det_eval = self.evaluator.evaluate_detection(
                analysis_results['detection']
            )
            evaluation_results['detection_metrics'] = det_eval
        
        # Overall performance analysis
        overall_performance = self.performance_analyzer.analyze_overall_performance(
            preprocessed_data, analysis_results
        )
        evaluation_results['overall_performance'] = overall_performance
        
        # Save evaluation report
        eval_report_path = output_path / 'evaluation_report.json'
        with open(eval_report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        return evaluation_results
    
    def _generate_visualizations(self,
                               preprocessed_data: Dict,
                               feature_data: Dict,
                               analysis_results: Dict,
                               output_path: Path) -> Dict:
        """Generate comprehensive visualizations."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        viz_results = {}
        image = preprocessed_data['image']
        
        # Original image visualization
        original_path = output_path / 'original_image.png'
        self.visualizer.plot_image(image, 'Original Satellite Image', str(original_path))
        viz_results['original_image'] = str(original_path)
        
        # Feature visualizations
        if 'comprehensive_features' in feature_data:
            features = feature_data['comprehensive_features']
            
            # Key feature maps
            key_features = ['vegetation_index', 'water_index', 'urban_index']
            for feature_name in key_features:
                if feature_name in features:
                    feat_path = output_path / f'{feature_name}.png'
                    self.visualizer.plot_heatmap(
                        features[feature_name],
                        f'{feature_name.replace("_", " ").title()}',
                        str(feat_path)
                    )
                    viz_results[feature_name] = str(feat_path)
        
        # Clustering visualization
        if 'cluster_map' in feature_data:
            cluster_path = output_path / 'land_cover_clusters.png'
            self.visualizer.plot_cluster_map(
                feature_data['cluster_map'],
                'Land Cover Classification',
                str(cluster_path)
            )
            viz_results['clustering'] = str(cluster_path)
        
        # Analysis-specific visualizations
        if 'segmentation' in analysis_results:
            seg_viz = self.satellite_visualizer.visualize_segmentation_results(
                image, analysis_results['segmentation'], output_path / 'segmentation'
            )
            viz_results['segmentation_viz'] = seg_viz
        
        if 'detection' in analysis_results:
            det_viz = self.satellite_visualizer.visualize_detection_results(
                image, analysis_results['detection'], output_path / 'detection'
            )
            viz_results['detection_viz'] = det_viz
        
        return viz_results
    
    def _save_results_summary(self, results: Dict, output_path: Path):
        """Save comprehensive results summary."""
        # Create summary report
        summary = {
            'analysis_info': results['input_info'],
            'processing_summary': {
                'total_processing_time': results['input_info']['processing_time'],
                'components_used': list(results.keys()),
                'output_files_generated': len(list(output_path.rglob('*.*')))
            },
            'key_findings': self._extract_key_findings(results)
        }
        
        # Save summary
        summary_path = output_path / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results summary saved to {summary_path}")
    
    def _extract_key_findings(self, results: Dict) -> Dict:
        """Extract key findings from analysis results."""
        findings = {}
        
        # Feature extraction findings
        if 'features' in results and 'cluster_info' in results['features']:
            cluster_info = results['features']['cluster_info']
            findings['land_cover_classes'] = cluster_info.get('n_clusters', 0)
        
        # Segmentation findings
        if 'analysis' in results and 'segmentation' in results['analysis']:
            seg_analysis = results['analysis']['segmentation'].get('land_cover_analysis', {})
            findings['segmentation_classes'] = len(seg_analysis.get('class_statistics', {}))
        
        # Detection findings
        if 'analysis' in results and 'detection' in results['analysis']:
            det_analysis = results['analysis']['detection'].get('detection_analysis', {})
            findings['objects_detected'] = det_analysis.get('total_detections', 0)
            findings['object_types'] = len(det_analysis.get('class_distribution', {}))
        
        return findings


def main():
    """Main entry point for GIS image analysis."""
    parser = argparse.ArgumentParser(description='GIS Image Analysis Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input TIFF image path')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--analysis-type', '-t', 
                       choices=['segmentation', 'detection', 'full'], 
                       default='full',
                       help='Type of analysis to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize pipeline
        pipeline = GISAnalysisPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_comprehensive_analysis(
            input_path=args.input,
            output_dir=args.output,
            analysis_type=args.analysis_type
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        print(f"⏱️  Processing time: {results['input_info']['processing_time']:.2f} seconds")
        
        # Print key findings
        if 'key_findings' in results.get('processing_summary', {}):
            findings = results['processing_summary']['key_findings']
            print(f"\n📋 Key Findings:")
            for key, value in findings.items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
    print(f"Task: {args.task}")
    
    try:
        # 1. Preprocessing
        print("\n1. Preprocessing image...")
        preprocessor = ImagePreprocessor(config.get('preprocessing', {}))
        processed_data = preprocessor.process_image(args.input)
        
        # 2. Feature Extraction
        print("\n2. Extracting features...")
        extractor = FeatureExtractor(config.get('feature_extraction', {}))
        features = extractor.extract_features(processed_data)
        
        # 3. Model Training/Inference
        print(f"\n3. Running {args.model} model...")
        trainer = ModelTrainer(
            model_type=args.model,
            task_type=args.task,
            config=config.get('model', {})
        )
        
        # Train or load model
        model = trainer.get_model()
        predictions = trainer.predict(processed_data)
        
        # 4. Evaluation
        print("\n4. Evaluating results...")
        evaluator = ModelEvaluator(config.get('evaluation', {}))
        metrics = evaluator.evaluate(predictions, ground_truth=None)
        
        # 5. Visualization
        print("\n5. Creating visualizations...")
        visualizer = Visualizer(output_dir=args.output)
        visualizer.create_land_cover_map(processed_data, predictions)
        visualizer.create_overlay_visualization(args.input, predictions)
        visualizer.save_metrics_plot(metrics)
        
        # Save results
        results_file = os.path.join(args.output, 'results.json')
        evaluator.save_results(metrics, results_file)
        
        print(f"\n✅ Analysis complete! Results saved to: {args.output}")
        print(f"📊 Metrics: {results_file}")
        print(f"🗺️  Visualizations: {args.output}/visualizations/")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
