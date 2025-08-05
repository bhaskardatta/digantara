"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """
    Configuration manager for loading and managing configuration files.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            # Create default config if file doesn't exist
            self._config = self._create_default_config()
            self.save_config(config_path)
        else:
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        
        return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if self._config is None:
            return self._create_default_config()
        return self._config
    
    def save_config(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(self._config or self._create_default_config(), f, default_flow_style=False, indent=2)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "preprocessing": {
                "target_size": [1024, 1024],
                "normalize": True,
                "enhance_contrast": True,
                "reduce_noise": True
            },
            "detection": {
                "model_size": "n",  # YOLOv8 nano for faster loading
                "confidence_threshold": 0.5,
                "iou_threshold": 0.5,
                "max_detections": 1000,
                "class_names": [
                    "building", "vehicle", "road", "bridge", "aircraft",
                    "ship", "storage_tank", "tower", "construction_site",
                    "sports_facility", "parking_lot", "residential_area"
                ]
            },
            "segmentation": {
                "num_classes": 6,
                "encoder_name": "resnet18",  # Lighter model for demo
                "encoder_weights": "imagenet",
                "activation": "softmax",
                "class_names": [
                    "background", "vegetation", "urban", "water", "agriculture", "bare_soil"
                ]
            },
            "evaluation": {
                "iou_thresholds": [0.5, 0.75],
                "confidence_threshold": 0.5
            },
            "visualization": {
                "dpi": 300,
                "figsize": [12, 8],
                "save_plots": True
            }
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
