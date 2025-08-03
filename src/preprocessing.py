"""
Image preprocessing utilities for GIS TIFF images.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, Optional, Union
import logging


class ImagePreprocessor:
    """Handles preprocessing of TIFF images for ML analysis."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.target_size = self.config.get('target_size', (512, 512))
        self.normalization_method = self.config.get('normalization', 'minmax')
        self.noise_reduction = self.config.get('noise_reduction', True)
        self.logger = logging.getLogger(__name__)
        
    def load_tiff(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Load TIFF image and metadata.
        
        Args:
            filepath: Path to TIFF file
            
        Returns:
            Tuple of (image_array, metadata)
        """
        try:
            with rasterio.open(filepath) as src:
                # Read all bands
                image = src.read()
                
                # Get metadata
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'bounds': src.bounds
                }
                
                # Convert to HWC format if multi-band
                if image.ndim == 3:
                    image = np.transpose(image, (1, 2, 0))
                
                self.logger.info(f"Loaded TIFF: {image.shape}, CRS: {metadata['crs']}")
                return image, metadata
                
        except Exception as e:
            self.logger.error(f"Error loading TIFF file: {str(e)}")
            raise
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image data quality.
        
        Args:
            image: Input image array
            
        Returns:
            True if image is valid
        """
        if image is None or image.size == 0:
            return False
            
        # Check for excessive NaN values
        nan_ratio = np.isnan(image).sum() / image.size
        if nan_ratio > 0.5:
            self.logger.warning(f"High NaN ratio: {nan_ratio:.2%}")
            
        # Check value ranges
        if image.ndim > 2:
            for band in range(image.shape[-1]):
                band_data = image[:, :, band]
                self.logger.info(f"Band {band}: min={band_data.min():.2f}, "
                               f"max={band_data.max():.2f}, "
                               f"mean={band_data.mean():.2f}")
        
        return True
    
    def handle_missing_data(self, image: np.ndarray) -> np.ndarray:
        """
        Handle missing or invalid data in the image.
        
        Args:
            image: Input image with potential missing data
            
        Returns:
            Image with missing data handled
        """
        # Handle NaN values
        if np.isnan(image).any():
            self.logger.info("Handling NaN values with interpolation")
            
            if image.ndim == 2:
                # For single band, use inpainting
                mask = np.isnan(image).astype(np.uint8)
                image_filled = cv2.inpaint(
                    image.astype(np.float32), 
                    mask, 
                    inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA
                )
                return image_filled
            else:
                # For multi-band, interpolate each band
                for band in range(image.shape[-1]):
                    band_data = image[:, :, band]
                    if np.isnan(band_data).any():
                        mask = np.isnan(band_data).astype(np.uint8)
                        image[:, :, band] = cv2.inpaint(
                            band_data.astype(np.float32),
                            mask,
                            inpaintRadius=3,
                            flags=cv2.INPAINT_TELEA
                        )
        
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        if image.shape[:2] != self.target_size:
            self.logger.info(f"Resizing from {image.shape[:2]} to {self.target_size}")
            
            if image.ndim == 2:
                resized = cv2.resize(image, self.target_size, 
                                   interpolation=cv2.INTER_CUBIC)
            else:
                resized = cv2.resize(image, self.target_size,
                                   interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if self.normalization_method == 'minmax':
            # Min-max normalization to [0, 1]
            scaler = MinMaxScaler()
            
            if image.ndim == 2:
                normalized = scaler.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
            else:
                # Normalize each band separately
                normalized = np.zeros_like(image, dtype=np.float32)
                for band in range(image.shape[-1]):
                    band_data = image[:, :, band].reshape(-1, 1)
                    normalized[:, :, band] = scaler.fit_transform(band_data).reshape(image.shape[:2])
            
            self.logger.info(f"Applied min-max normalization")
            return normalized.astype(np.float32)
            
        elif self.normalization_method == 'standard':
            # Z-score normalization
            scaler = StandardScaler()
            
            if image.ndim == 2:
                normalized = scaler.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
            else:
                normalized = np.zeros_like(image, dtype=np.float32)
                for band in range(image.shape[-1]):
                    band_data = image[:, :, band].reshape(-1, 1)
                    normalized[:, :, band] = scaler.fit_transform(band_data).reshape(image.shape[:2])
            
            self.logger.info(f"Applied standard normalization")
            return normalized.astype(np.float32)
        
        return image.astype(np.float32)
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction filters.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        if not self.noise_reduction:
            return image
        
        if image.ndim == 2:
            # Gaussian blur for single band
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
        else:
            # Apply to each band
            denoised = np.zeros_like(image)
            for band in range(image.shape[-1]):
                denoised[:, :, band] = cv2.GaussianBlur(image[:, :, band], (3, 3), 0)
        
        self.logger.info("Applied noise reduction")
        return denoised
    
    def process_image(self, filepath: str) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to input TIFF file
            
        Returns:
            Dictionary containing processed image and metadata
        """
        self.logger.info(f"Starting preprocessing pipeline for: {filepath}")
        
        # Load image
        image, metadata = self.load_tiff(filepath)
        
        # Validate
        if not self.validate_image(image):
            raise ValueError("Image validation failed")
        
        # Handle missing data
        image = self.handle_missing_data(image)
        
        # Resize if needed
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Reduce noise
        image = self.reduce_noise(image)
        
        result = {
            'image': image,
            'metadata': metadata,
            'original_shape': (metadata['width'], metadata['height']),
            'processed_shape': image.shape,
            'preprocessing_config': self.config
        }
        
        self.logger.info("Preprocessing completed successfully")
        return result
