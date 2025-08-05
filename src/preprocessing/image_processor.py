"""
Image preprocessing utilities for TIFF files
"""

import numpy as np
import rasterio
import cv2
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import logging
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling


class ImageProcessor:
    """
    Handles loading, preprocessing, and normalization of TIFF images.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize image processor.
        
        Args:
            config: Preprocessing configuration
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.target_size = config.get("target_size", (1024, 1024))
        self.normalize = config.get("normalize", True)
        self.enhance_contrast = config.get("enhance_contrast", True)
        self.remove_clouds = config.get("remove_clouds", False)
        
    def load_and_preprocess(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and preprocess TIFF image.
        
        Args:
            image_path: Path to TIFF file
            
        Returns:
            Preprocessed image data dictionary
        """
        image_path = Path(image_path)
        self.logger.info(f"Loading image: {image_path}")
        
        with rasterio.open(image_path) as src:
            # Read image metadata
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else "unknown",
                "crs": str(src.crs) if src.crs else None,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata
            }
            
            # Read image data
            image = src.read()  # Shape: (bands, height, width)
            
        self.logger.info(f"Loaded image with shape: {image.shape}, dtype: {image.dtype}")
        
        # Preprocess image
        processed_data = self._preprocess_image(image, metadata)
        
        return {
            "raw": image,
            "rgb": processed_data["rgb"],
            "normalized": processed_data["normalized"],
            "metadata": metadata,
            "preprocessing_info": processed_data["info"]
        }
        
    def _preprocess_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply preprocessing steps to image.
        
        Args:
            image: Raw image array
            metadata: Image metadata
            
        Returns:
            Preprocessed image data
        """
        preprocessing_info = {}
        
        # Handle different band configurations
        rgb_image = self._extract_rgb_bands(image, metadata)
        preprocessing_info["rgb_bands"] = "RGB extracted"
        
        # Handle nodata values
        if metadata["nodata"] is not None:
            rgb_image = self._handle_nodata(rgb_image, metadata["nodata"])
            preprocessing_info["nodata_handled"] = True
            
        # Normalize pixel values
        if self.normalize:
            rgb_image = self._normalize_image(rgb_image)
            preprocessing_info["normalized"] = True
            
        # Enhance contrast
        if self.enhance_contrast:
            rgb_image = self._enhance_contrast(rgb_image)
            preprocessing_info["contrast_enhanced"] = True
            
        # Resize image
        original_shape = rgb_image.shape[:2]
        rgb_image = self._resize_image(rgb_image, self.target_size)
        preprocessing_info["resized"] = f"{original_shape} -> {rgb_image.shape[:2]}"
        
        # Apply noise reduction
        rgb_image = self._reduce_noise(rgb_image)
        preprocessing_info["noise_reduced"] = True
        
        # Convert to tensor
        rgb_tensor = self._to_tensor(rgb_image)
        
        return {
            "rgb": rgb_tensor,
            "normalized": rgb_image,
            "info": preprocessing_info
        }
        
    def _extract_rgb_bands(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract RGB bands from multi-band image."""
        bands = image.shape[0]
        
        if bands == 1:
            # Grayscale - convert to RGB
            gray = image[0]
            rgb = np.stack([gray, gray, gray], axis=2)
        elif bands >= 3:
            # Multi-band - take first 3 as RGB
            rgb = np.transpose(image[:3], (1, 2, 0))
        else:
            raise ValueError(f"Unsupported number of bands: {bands}")
            
        return rgb.astype(np.float32)
        
    def _handle_nodata(self, image: np.ndarray, nodata_value: float) -> np.ndarray:
        """Handle nodata values by interpolation or masking."""
        mask = image == nodata_value
        if np.any(mask):
            # Simple interpolation - replace with local mean
            for i in range(image.shape[2]):
                band = image[:, :, i]
                band_mask = mask[:, :, i]
                if np.any(band_mask):
                    # Replace with mean of non-nodata pixels
                    valid_pixels = band[~band_mask]
                    if len(valid_pixels) > 0:
                        band[band_mask] = np.mean(valid_pixels)
                        
        return image
        
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1] range."""
        # Handle potential outliers by using percentile-based normalization
        for i in range(image.shape[2]):
            band = image[:, :, i]
            p1, p99 = np.percentile(band, [1, 99])
            band = np.clip(band, p1, p99)
            image[:, :, i] = (band - p1) / (p99 - p1) if p99 > p1 else band
            
        return np.clip(image, 0, 1)
        
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        # Convert to uint8 for CLAHE
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(image_uint8.shape[2]):
            image_uint8[:, :, i] = clahe.apply(image_uint8[:, :, i])
            
        # Convert back to float32
        return image_uint8.astype(np.float32) / 255.0
        
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving important details for large aerial images."""
        # Use high-quality interpolation for large downsampling
        if image.shape[0] > target_size[0] * 4 or image.shape[1] > target_size[1] * 4:
            # For very large images, do progressive downsampling to preserve details
            current_image = image.copy()
            while current_image.shape[0] > target_size[0] * 2 or current_image.shape[1] > target_size[1] * 2:
                new_height = max(target_size[0], current_image.shape[0] // 2)
                new_width = max(target_size[1], current_image.shape[1] // 2)
                current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Final resize with high-quality interpolation
            resized = cv2.resize(current_image, target_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            # Standard resize for smaller images
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
        return resized
        
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction filter."""
        # Convert to uint8 for filtering
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image_uint8, 9, 75, 75)
        
        return filtered.astype(np.float32) / 255.0
        
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        # Convert HWC to CHW format
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image_tensor.to(self.device)
        
    def calculate_ndvi(self, nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Args:
            nir_band: Near-infrared band
            red_band: Red band
            
        Returns:
            NDVI array
        """
        # Avoid division by zero
        denominator = nir_band + red_band
        denominator[denominator == 0] = 1e-8
        
        ndvi = (nir_band - red_band) / denominator
        return np.clip(ndvi, -1, 1)
        
    def calculate_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate texture features using GLCM.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary of texture features
        """
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to uint8
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
            
        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        glcm = graycomatrix(
            image_uint8,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Calculate texture properties
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        
        for prop in properties:
            features[prop] = graycoprops(glcm, prop).mean()
            
        return features
