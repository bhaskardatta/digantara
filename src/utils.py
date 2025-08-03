"""
Utility functions for GIS image analysis pipeline.
Enhanced with satellite imagery processing utilities.
"""

import logging
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
from datetime import datetime


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None, verbose: bool = False):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        verbose: Enable verbose logging (overrides log_level to DEBUG)
    """
    if verbose:
        log_level = 'DEBUG'
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    
    if log_file:
        logger.info(f"Log file: {log_file}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path or not Path(config_path).exists():
        logger = logging.getLogger(__name__)
        logger.warning(f"Config file not found: {config_path}. Using default configuration.")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the GIS analysis pipeline.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'preprocessing': {
            'target_size': [512, 512],
            'normalize': True,
            'handle_missing_data': True,
            'noise_reduction': True,
            'output_format': 'uint8'
        },
        'feature_extraction': {
            'tile_size': 256,
            'overlap': 0.25,
            'n_clusters': 5,
            'feature_type': 'hybrid',
            'use_deep_features': True
        },
        'segformer': {
            'model_name': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'confidence_threshold': 0.5,
            'tile_size': 512,
            'overlap': 0.25,
            'land_cover_classes': [
                'background', 'urban', 'forest', 'water', 'agriculture', 'other'
            ]
        },
        'yolo': {
            'model_name': 'yolo_world',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'tile_size': 640,
            'overlap': 0.2,
            'satellite_classes': [
                'building', 'road', 'vehicle', 'tree', 'water', 'field'
            ]
        },
        'evaluation': {
            'metrics': ['iou', 'f1_score', 'precision', 'recall'],
            'save_confusion_matrix': True,
            'generate_report': True
        },
        'visualization': {
            'save_format': 'png',
            'dpi': 300,
            'colormap': 'viridis',
            'figure_size': [12, 8]
        },
        'satellite_viz': {
            'land_cover_colors': {
                0: [0, 0, 0],      # background - black
                1: [255, 0, 0],    # urban - red
                2: [0, 255, 0],    # forest - green
                3: [0, 0, 255],    # water - blue
                4: [255, 255, 0],  # agriculture - yellow
                5: [128, 128, 128] # other - gray
            },
            'detection_colors': {
                'building': [255, 0, 0],
                'road': [128, 128, 128],
                'vehicle': [255, 255, 0],
                'tree': [0, 255, 0],
                'water': [0, 0, 255],
                'field': [255, 165, 0]
            }
        }
    }


def create_output_structure(output_path: Path):
    """
    Create comprehensive output directory structure.
    
    Args:
        output_path: Base output directory path
    """
    # Define subdirectories
    subdirs = [
        'preprocessing',
        'features',
        'segmentation',
        'detection',
        'combined',
        'evaluation',
        'visualizations',
        'visualizations/segmentation',
        'visualizations/detection',
        'visualizations/features',
        'logs',
        'reports'
    ]
    
    # Create directories
    for subdir in subdirs:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Output structure created at {output_path}")


def save_results_json(data: Dict[str, Any], filepath: Union[str, Path]):
    """
    Save results to JSON file with proper serialization.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    def json_serializer(obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return {
                'type': 'numpy_array',
                'data': obj.tolist(),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save results: {e}")


def load_results_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from JSON file with proper deserialization.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data dictionary
    """
    def json_deserializer(data):
        """Custom JSON deserializer for numpy arrays."""
        if isinstance(data, dict) and data.get('type') == 'numpy_array':
            return np.array(data['data']).reshape(data['shape']).astype(data['dtype'])
        return data
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Recursively deserialize numpy arrays
        def recursive_deserialize(obj):
            if isinstance(obj, dict):
                if obj.get('type') == 'numpy_array':
                    return np.array(obj['data']).reshape(obj['shape']).astype(obj['dtype'])
                else:
                    return {k: recursive_deserialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_deserialize(item) for item in obj]
            else:
                return obj
        
        return recursive_deserialize(data)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load results: {e}")
        return {}


def validate_input_image(image_path: str) -> bool:
    """
    Validate input image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    # Check file extension
    valid_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    file_ext = Path(image_path).suffix.lower()
    
    return file_ext in valid_extensions


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary containing image information
    """
    try:
        from PIL import Image
        import os
        
        # Basic file info
        file_stats = os.stat(image_path)
        
        # Image info
        with Image.open(image_path) as img:
            info = {
                'filepath': image_path,
                'filename': Path(image_path).name,
                'file_size_mb': file_stats.st_size / (1024 * 1024),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'channels': len(img.getbands()) if hasattr(img, 'getbands') else 1,
                'bands': img.getbands() if hasattr(img, 'getbands') else None,
                'created': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
            
            # Try to get additional TIFF info
            if img.format == 'TIFF':
                info['tiff_info'] = {}
                for key, value in img.tag_v2.items():
                    try:
                        info['tiff_info'][str(key)] = str(value)
                    except:
                        pass
        
        return info
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get image info: {e}")
        return {'filepath': image_path, 'error': str(e)}


def calculate_processing_stats(start_time: float, end_time: float, 
                             image_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate processing statistics.
    
    Args:
        start_time: Processing start time
        end_time: Processing end time
        image_info: Image information dictionary
        
    Returns:
        Processing statistics dictionary
    """
    processing_time = end_time - start_time
    
    stats = {
        'processing_time_seconds': processing_time,
        'processing_time_formatted': f"{processing_time:.2f}s",
        'timestamp': datetime.now().isoformat(),
        'pixels_processed': image_info.get('width', 0) * image_info.get('height', 0),
        'megapixels': (image_info.get('width', 0) * image_info.get('height', 0)) / 1_000_000,
        'processing_rate_mpx_per_second': (image_info.get('width', 0) * image_info.get('height', 0)) / 1_000_000 / processing_time if processing_time > 0 else 0
    }
    
    return stats


def create_analysis_report(results: Dict[str, Any], output_path: Path):
    """
    Create comprehensive analysis report.
    
    Args:
        results: Analysis results dictionary
        output_path: Output directory path
    """
    report_content = f"""
# GIS Image Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Information
- **Image Path:** {results.get('input_info', {}).get('input_path', 'N/A')}
- **Analysis Type:** {results.get('input_info', {}).get('analysis_type', 'N/A')}
- **Processing Time:** {results.get('input_info', {}).get('processing_time', 0):.2f} seconds

## Processing Summary
"""
    
    # Add processing details
    if 'preprocessing' in results:
        preprocessing = results['preprocessing']
        report_content += f"""
### Preprocessing
- **Image Shape:** {preprocessing.get('image', np.array([])).shape}
- **Data Type:** {preprocessing.get('image', np.array([])).dtype}
- **Normalization:** Applied
"""
    
    # Add feature extraction details
    if 'features' in results:
        features = results['features']
        cluster_info = features.get('cluster_info', {})
        report_content += f"""
### Feature Extraction
- **Land Cover Classes:** {cluster_info.get('n_clusters', 'N/A')}
- **Feature Types:** {len(features.get('comprehensive_features', {}))}
"""
    
    # Add analysis results
    if 'analysis' in results:
        analysis = results['analysis']
        
        if 'segmentation' in analysis:
            seg_analysis = analysis['segmentation'].get('land_cover_analysis', {})
            report_content += f"""
### Segmentation Results
- **Classes Detected:** {len(seg_analysis.get('class_statistics', {}))}
"""
        
        if 'detection' in analysis:
            det_analysis = analysis['detection'].get('detection_analysis', {})
            report_content += f"""
### Detection Results
- **Objects Detected:** {det_analysis.get('total_detections', 0)}
- **Object Types:** {len(det_analysis.get('class_distribution', {}))}
"""
    
    # Save report
    report_path = output_path / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Analysis report saved to {report_path}")


def normalize_array(array: np.ndarray, target_range: tuple = (0, 1)) -> np.ndarray:
    """
    Normalize array to target range.
    
    Args:
        array: Input array
        target_range: Target range (min, max)
        
    Returns:
        Normalized array
    """
    min_val, max_val = target_range
    
    # Handle constant arrays
    if np.ptp(array) == 0:
        return np.full_like(array, min_val)
    
    # Normalize to 0-1 first
    normalized = (array - np.min(array)) / (np.max(array) - np.min(array))
    
    # Scale to target range
    return normalized * (max_val - min_val) + min_val


def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    """
    Safe division with default value for division by zero.
    
    Args:
        a: Numerator array
        b: Denominator array
        default: Default value for division by zero
        
    Returns:
        Result of safe division
    """
    return np.divide(a, b, out=np.full_like(a, default, dtype=float), where=(b != 0))


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in RGB format.
    
    Args:
        image: Input image array
        
    Returns:
        RGB image array
    """
    if image.ndim == 2:
        # Grayscale to RGB
        return np.stack([image, image, image], axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        # Single channel to RGB
        return np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 3:
        # Already RGB
        return image
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA to RGB
        return image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Memory usage dictionary
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, step: int = None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = (self.current_step / self.total_steps) * 100
        self.logger.info(f"{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps})")
    
    def finish(self):
        """Mark as finished."""
        self.current_step = self.total_steps
        self.logger.info(f"{self.description}: Completed!")


# Export commonly used functions
__all__ = [
    'setup_logging',
    'load_config', 
    'get_default_config',
    'create_output_structure',
    'save_results_json',
    'load_results_json',
    'validate_input_image',
    'get_image_info',
    'calculate_processing_stats',
    'create_analysis_report',
    'normalize_array',
    'safe_divide',
    'ensure_rgb',
    'get_memory_usage',
    'ProgressTracker'
]

import logging
import json
import os
import sys
from typing import Dict, Optional, Any
from pathlib import Path


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging configured successfully")


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict:
    """
    Get default configuration settings.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'preprocessing': {
            'target_size': (512, 512),
            'normalization': 'minmax',
            'noise_reduction': True
        },
        'feature_extraction': {
            'method': 'clustering',
            'n_clusters': 5,
            'feature_type': 'spectral'
        },
        'model': {
            'num_classes': 5,
            'in_channels': 3,
            'pretrained': True,
            'input_shape': (512, 512, 3)
        },
        'evaluation': {
            'calculate_spatial_metrics': True
        }
    }


def save_config(config: Dict, config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Output file path
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Configuration saved to: {config_path}")
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")


def validate_input_file(filepath: str) -> bool:
    """
    Validate input TIFF file.
    
    Args:
        filepath: Path to input file
        
    Returns:
        True if file is valid
    """
    if not os.path.exists(filepath):
        logging.error(f"Input file does not exist: {filepath}")
        return False
    
    if not filepath.lower().endswith(('.tif', '.tiff')):
        logging.warning(f"File may not be a TIFF: {filepath}")
    
    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        logging.error(f"Input file is empty: {filepath}")
        return False
    
    logging.info(f"Input file validated: {filepath} ({file_size} bytes)")
    return True


def create_output_structure(output_dir: str):
    """
    Create output directory structure.
    
    Args:
        output_dir: Base output directory
    """
    subdirs = [
        'visualizations',
        'results',
        'models',
        'reports'
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(output_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
    
    logging.info(f"Output structure created: {output_dir}")


def format_memory_size(size_bytes: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_system_info() -> Dict:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': format_memory_size(psutil.virtual_memory().total),
        'memory_available': format_memory_size(psutil.virtual_memory().available)
    }


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Returns:
        Dictionary showing which dependencies are available
    """
    dependencies = {
        'numpy': False,
        'rasterio': False,
        'opencv': False,
        'sklearn': False,
        'torch': False,
        'tensorflow': False,
        'matplotlib': False,
        'seaborn': False,
        'geopandas': False
    }
    
    # Check each dependency
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import rasterio
        dependencies['rasterio'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        dependencies['tensorflow'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        pass
    
    try:
        import seaborn
        dependencies['seaborn'] = True
    except ImportError:
        pass
    
    try:
        import geopandas
        dependencies['geopandas'] = True
    except ImportError:
        pass
    
    return dependencies


def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_dependencies()
    
    print("\nDependency Status:")
    print("-" * 30)
    
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{dep:12} {status}")
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")


def calculate_processing_time(start_time: float, end_time: float) -> str:
    """
    Calculate and format processing time.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Formatted time string
    """
    duration = end_time - start_time
    
    if duration < 60:
        return f"{duration:.2f} seconds"
    elif duration < 3600:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_sample_data(output_dir: str):
    """
    Create sample data for testing (when no real TIFF is available).
    
    Args:
        output_dir: Directory to save sample data
    """
    try:
        import numpy as np
        
        # Create synthetic satellite-like image
        height, width = 512, 512
        channels = 3
        
        # Generate base terrain
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Create different land cover patterns
        water = (np.sin(X) * np.cos(Y) + np.random.normal(0, 0.1, (height, width))) > 0.8
        forest = (np.sin(X * 2) * np.cos(Y * 2) + np.random.normal(0, 0.2, (height, width))) > 0.5
        urban = (np.abs(X - 5) < 2) & (np.abs(Y - 5) < 2) & (np.random.rand(height, width) > 0.3)
        
        # Create RGB image
        image = np.zeros((height, width, channels))
        
        # Water - blue
        image[water] = [0.2, 0.4, 0.8]
        
        # Forest - green
        image[forest & ~water] = [0.2, 0.6, 0.2]
        
        # Urban - gray/red
        image[urban & ~water & ~forest] = [0.6, 0.5, 0.4]
        
        # Agriculture/bare soil - brown/yellow
        remaining = ~(water | forest | urban)
        image[remaining] = [0.8, 0.7, 0.4]
        
        # Add some noise
        image += np.random.normal(0, 0.05, image.shape)
        image = np.clip(image, 0, 1)
        
        # Save as numpy array (since we can't guarantee rasterio)
        sample_dir = os.path.join(output_dir, 'data', 'raw')
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_file = os.path.join(sample_dir, 'sample_image.npy')
        np.save(sample_file, image)
        
        logging.info(f"Sample data created: {sample_file}")
        return sample_file
        
    except ImportError:
        logging.warning("Cannot create sample data - numpy not available")
        return None


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the process
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def update(self, step: int = None):
        """
        Update progress.
        
        Args:
            step: Current step number (optional)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        percentage = (self.current_step / self.total_steps) * 100
        print(f"\r{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps})", 
              end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete


def validate_model_output(predictions: Any, expected_shape: tuple) -> bool:
    """
    Validate model output format.
    
    Args:
        predictions: Model predictions
        expected_shape: Expected shape of predictions
        
    Returns:
        True if output is valid
    """
    try:
        import numpy as np
        
        if not isinstance(predictions, np.ndarray):
            logging.error("Predictions must be numpy array")
            return False
        
        if predictions.shape != expected_shape:
            logging.error(f"Shape mismatch: got {predictions.shape}, expected {expected_shape}")
            return False
        
        # Check for valid class indices
        unique_values = np.unique(predictions)
        if np.any(unique_values < 0):
            logging.error("Predictions contain negative values")
            return False
        
        logging.info("Model output validation passed")
        return True
        
    except ImportError:
        logging.warning("Cannot validate model output - numpy not available")
        return True  # Assume valid if we can't check
