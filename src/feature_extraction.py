"""
Advanced feature extraction methods for satellite imagery using deep learning.
Integrates with SegFormer and YOLO-World for comprehensive analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import cv2
from pathlib import Path
from PIL import Image
import json

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import SegformerImageProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/Transformers not available. Deep learning features disabled.")

# Traditional ML imports
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some features disabled.")


class SatelliteFeatureExtractor:
    """
    Advanced feature extractor for satellite imagery using multiple methods:
    - Deep learning features (SegFormer backbone)
    - Traditional computer vision features
    - Spectral and textural analysis
    - Multi-scale feature pyramid
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize advanced feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.tile_size = self.config.get('tile_size', 256)
        self.overlap = self.config.get('overlap', 0.25)
        self.n_clusters = self.config.get('n_clusters', 5)
        self.feature_type = self.config.get('feature_type', 'hybrid')
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize deep learning components
        self.segformer_processor = None
        self.feature_extractor_model = None
        
        if TORCH_AVAILABLE:
            self._initialize_deep_learning_components()
        
        # Initialize traditional feature extractors
        if SKLEARN_AVAILABLE:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.pca = PCA(n_components=min(50, self.n_clusters * 10))
            self.scaler = StandardScaler()
        
        self.logger.info("SatelliteFeatureExtractor initialized")
    
    def _initialize_deep_learning_components(self):
        """Initialize deep learning feature extraction components."""
        try:
            # Load SegFormer processor for consistent preprocessing
            self.segformer_processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            
            # Create a lightweight feature extractor based on SegFormer encoder
            self.feature_extractor_model = SatelliteFeatureBackbone()
            
            self.logger.info("Deep learning components initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize deep learning components: {e}")
            self.segformer_processor = None
            self.feature_extractor_model = None
    
    def extract_comprehensive_features(self, 
                                     image: np.ndarray,
                                     use_deep_features: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features from satellite image.
        
        Args:
            image: Input satellite image (H, W, C)
            use_deep_features: Whether to use deep learning features
            
        Returns:
            Dictionary containing all extracted features
        """
        self.logger.info("Starting comprehensive feature extraction...")
        
        features = {}
        
        # 1. Extract traditional computer vision features
        features.update(self._extract_traditional_features(image))
        
        # 2. Extract spectral features
        features.update(self._extract_spectral_features(image))
        
        # 3. Extract texture features
        features.update(self._extract_texture_features(image))
        
        # 4. Extract morphological features
        features.update(self._extract_morphological_features(image))
        
        # 5. Extract deep learning features (if available)
        if use_deep_features and self.feature_extractor_model is not None:
            deep_features = self._extract_deep_features(image)
            features.update(deep_features)
        
        # 6. Create multi-scale feature pyramid
        features['multi_scale_features'] = self._create_feature_pyramid(image)
        
        self.logger.info(f"Feature extraction completed. {len(features)} feature types extracted.")
        
        return features
    
    def _extract_traditional_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract traditional computer vision features."""
        features = {}
        
        # Convert to grayscale for some operations
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Edge features
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        features['edge_magnitude'] = gradient_magnitude
        features['edge_direction'] = gradient_direction
        
        # Corner features (Harris corners)
        corners = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
        features['corners'] = corners
        
        # Blob detection using Difference of Gaussians
        blur1 = cv2.GaussianBlur(gray.astype(np.float32), (9, 9), 1.0)
        blur2 = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 2.0)
        dog = blur1 - blur2
        features['blob_response'] = dog
        
        return features
    
    def _extract_spectral_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features from multi-band image."""
        features = {}
        
        if image.ndim == 2:
            # Single band - create dummy spectral features
            features['spectral_intensity'] = image
            return features
        
        h, w, c = image.shape
        
        # Individual band statistics
        for i in range(c):
            band = image[:, :, i].astype(np.float32)
            features[f'band_{i}_intensity'] = band
            
            # Band statistics (local)
            kernel = np.ones((5, 5), np.float32) / 25
            band_mean = cv2.filter2D(band, -1, kernel)
            band_var = cv2.filter2D(band**2, -1, kernel) - band_mean**2
            
            features[f'band_{i}_local_mean'] = band_mean
            features[f'band_{i}_local_variance'] = band_var
        
        # Spectral indices (if RGB or multispectral)
        if c >= 3:
            red = image[:, :, 0].astype(np.float32)
            green = image[:, :, 1].astype(np.float32)
            blue = image[:, :, 2].astype(np.float32)
            
            # Vegetation indices
            ndvi_like = (green - red) / (green + red + 1e-8)
            features['vegetation_index'] = ndvi_like
            
            # Water index
            water_index = (blue - green) / (blue + green + 1e-8)
            features['water_index'] = water_index
            
            # Urban index (using red and blue)
            urban_index = (red - blue) / (red + blue + 1e-8)
            features['urban_index'] = urban_index
            
            # Brightness
            brightness = np.mean(image, axis=2)
            features['brightness'] = brightness
            
            # Color ratios
            features['red_green_ratio'] = red / (green + 1e-8)
            features['red_blue_ratio'] = red / (blue + 1e-8)
            features['green_blue_ratio'] = green / (blue + 1e-8)
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract texture features using various methods."""
        features = {}
        
        # Convert to grayscale for texture analysis
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Local Binary Pattern (simplified version)
        lbp = self._compute_lbp(gray)
        features['lbp'] = lbp
        
        # Gradient-based texture
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Local texture statistics
        kernel = np.ones((7, 7), np.float32)
        
        # Local standard deviation
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel / kernel.sum())
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel / kernel.sum())
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        features['local_std'] = local_std
        
        # Local range
        kernel_max = cv2.dilate(gray, kernel.astype(np.uint8))
        kernel_min = cv2.erode(gray, kernel.astype(np.uint8))
        local_range = kernel_max - kernel_min
        features['local_range'] = local_range.astype(np.float32)
        
        # Gabor filter responses (simplified)
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            gabor_kernel = cv2.getGaborKernel((15, 15), sigma=3, theta=np.radians(theta), 
                                            lambd=10, gamma=0.5)
            gabor_response = cv2.filter2D(gray.astype(np.float32), cv2.CV_8UC3, gabor_kernel)
            gabor_responses.append(gabor_response)
        
        features['gabor_0'] = gabor_responses[0]
        features['gabor_45'] = gabor_responses[1]
        features['gabor_90'] = gabor_responses[2]
        features['gabor_135'] = gabor_responses[3]
        
        return features
    
    def _compute_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern (simplified version)."""
        h, w = gray_image.shape
        lbp = np.zeros_like(gray_image, dtype=np.float32)
        
        # 3x3 neighborhood LBP
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                code = 0
                
                # Check 8 neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp / 255.0  # Normalize
    
    def _extract_morphological_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract morphological features."""
        features = {}
        
        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Define morphological kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Basic morphological operations
        erosion = cv2.erode(gray, kernel_medium, iterations=1)
        dilation = cv2.dilate(gray, kernel_medium, iterations=1)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_medium)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_medium)
        
        features['erosion'] = erosion.astype(np.float32) / 255.0
        features['dilation'] = dilation.astype(np.float32) / 255.0
        features['opening'] = opening.astype(np.float32) / 255.0
        features['closing'] = closing.astype(np.float32) / 255.0
        
        # Top-hat and black-hat transforms
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_large)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_large)
        
        features['tophat'] = tophat.astype(np.float32) / 255.0
        features['blackhat'] = blackhat.astype(np.float32) / 255.0
        
        # Morphological gradient
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_small)
        features['morphological_gradient'] = gradient.astype(np.float32) / 255.0
        
        return features
    
    def _extract_deep_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract deep learning features using SegFormer backbone."""
        if not TORCH_AVAILABLE or self.feature_extractor_model is None:
            return {}
        
        features = {}
        
        try:
            # Preprocess image for SegFormer
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            
            # Process with SegFormer processor
            processed = self.segformer_processor(pil_image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            
            # Extract features using the backbone
            with torch.no_grad():
                deep_features = self.feature_extractor_model(pixel_values)
            
            # Convert to numpy and add to features
            for level, feat in deep_features.items():
                feat_np = feat.squeeze(0).cpu().numpy()  # Remove batch dimension
                features[f'deep_{level}'] = feat_np
            
            self.logger.info("Deep features extracted successfully")
            
        except Exception as e:
            self.logger.warning(f"Deep feature extraction failed: {e}")
        
        return features
    
    def _create_feature_pyramid(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create multi-scale feature pyramid."""
        pyramid = {}
        
        scales = [1.0, 0.5, 0.25]  # Different scales
        
        for i, scale in enumerate(scales):
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_image = image
            
            # Extract basic features at this scale
            if scaled_image.ndim == 3:
                gray = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = scaled_image
            
            # Compute gradient magnitude at this scale
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Resize back to original size
            if scale != 1.0:
                h_orig, w_orig = image.shape[:2]
                gradient_mag = cv2.resize(gradient_mag, (w_orig, h_orig), 
                                        interpolation=cv2.INTER_LINEAR)
            
            pyramid[f'scale_{i}'] = gradient_mag
        
        return pyramid
    
    def extract_features(self, processed_data: Dict) -> Dict:
        """
        Main feature extraction pipeline (backward compatibility).
        
        Args:
            processed_data: Dictionary containing preprocessed image and metadata
            
        Returns:
            Dictionary containing extracted features and analysis results
        """
        image = processed_data['image']
        
        self.logger.info("Starting feature extraction pipeline...")
        
        # Extract comprehensive features
        all_features = self.extract_comprehensive_features(image)
        
        # Create feature vectors for clustering
        feature_vectors = self._create_feature_vectors(all_features)
        
        # Perform clustering for land cover classification
        cluster_map, cluster_info = self._perform_advanced_clustering(feature_vectors, image.shape[:2])
        
        result = {
            'comprehensive_features': all_features,
            'feature_vectors': feature_vectors,
            'cluster_map': cluster_map,
            'cluster_info': cluster_info,
            'feature_extraction_config': self.config
        }
        
        self.logger.info("Feature extraction completed successfully")
        return result
    
    def _create_feature_vectors(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Create feature vectors from extracted features for clustering."""
        height, width = None, None
        
        # Determine image dimensions
        for key, feature in features.items():
            if isinstance(feature, np.ndarray) and feature.ndim >= 2:
                height, width = feature.shape[:2]
                break
        
        if height is None or width is None:
            raise ValueError("Could not determine image dimensions from features")
        
        # Collect all features into vectors
        feature_list = []
        
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                if feature.ndim == 2:
                    # Single channel feature
                    feature_list.append(feature.flatten())
                elif feature.ndim == 3:
                    # Multi-channel feature
                    for c in range(feature.shape[2]):
                        feature_list.append(feature[:, :, c].flatten())
        
        # Stack all features
        if feature_list:
            feature_matrix = np.stack(feature_list, axis=1)  # (n_pixels, n_features)
        else:
            # Fallback: use original image
            image = features.get('brightness', np.random.rand(height, width))
            feature_matrix = image.flatten().reshape(-1, 1)
        
        # Apply PCA if too many features
        if SKLEARN_AVAILABLE and feature_matrix.shape[1] > 50:
            self.logger.info(f"Applying PCA: {feature_matrix.shape[1]} -> 50 features")
            feature_matrix = self.pca.fit_transform(feature_matrix)
        
        return feature_matrix
    
    def _perform_advanced_clustering(self, 
                                   feature_vectors: np.ndarray, 
                                   image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """Perform advanced clustering on feature vectors."""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple clustering
            return self._simple_clustering_fallback(feature_vectors, image_shape)
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(feature_vectors)
        
        # Apply K-means clustering
        cluster_labels = self.kmeans.fit_transform(normalized_features)
        
        # Reshape to image dimensions
        cluster_map = cluster_labels.reshape(image_shape)
        
        # Calculate cluster statistics
        cluster_info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': self.kmeans.cluster_centers_,
            'inertia': self.kmeans.inertia_,
            'feature_importance': self._calculate_feature_importance(normalized_features, cluster_labels)
        }
        
        return cluster_map, cluster_info
    
    def _simple_clustering_fallback(self, 
                                  feature_vectors: np.ndarray, 
                                  image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """Simple clustering fallback when sklearn is not available."""
        # Simple k-means implementation
        n_pixels = feature_vectors.shape[0]
        
        # Initialize cluster centers randomly
        cluster_centers = []
        for _ in range(self.n_clusters):
            center_idx = np.random.randint(0, n_pixels)
            cluster_centers.append(feature_vectors[center_idx])
        
        cluster_centers = np.array(cluster_centers)
        
        # Assign pixels to clusters (simple distance-based)
        cluster_labels = np.zeros(n_pixels)
        
        for i, pixel_features in enumerate(feature_vectors):
            distances = [np.linalg.norm(pixel_features - center) for center in cluster_centers]
            cluster_labels[i] = np.argmin(distances)
        
        cluster_map = cluster_labels.reshape(image_shape)
        
        cluster_info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': cluster_centers,
            'inertia': 0.0  # Not calculated in simple version
        }
        
        return cluster_map, cluster_info
    
    def _calculate_feature_importance(self, 
                                    features: np.ndarray, 
                                    labels: np.ndarray) -> List[float]:
        """Calculate feature importance for clustering."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Train a random forest to determine feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(features, labels)
            
            return rf.feature_importances_.tolist()
            
        except ImportError:
            # Return uniform importance if sklearn RF not available
            return [1.0 / features.shape[1]] * features.shape[1]


if TORCH_AVAILABLE:
    class SatelliteFeatureBackbone(nn.Module):
        """
        Lightweight feature extraction backbone based on SegFormer encoder.
        Extracts multi-scale features for satellite imagery analysis.
        """
        
        def __init__(self):
            super().__init__()
            
            # Simple CNN backbone for feature extraction
            self.feature_pyramid = nn.ModuleDict({
                'layer1': self._make_layer(3, 64, 2),      # 1/2 scale
                'layer2': self._make_layer(64, 128, 2),    # 1/4 scale  
                'layer3': self._make_layer(128, 256, 2),   # 1/8 scale
                'layer4': self._make_layer(256, 512, 2)    # 1/16 scale
            })
        
        def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
            """Create a feature extraction layer."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Extract multi-scale features."""
            features = {}
            
            current = x
            for name, layer in self.feature_pyramid.items():
                current = layer(current)
                features[name] = current
            
            return features
else:
    # Dummy class when PyTorch is not available
    class SatelliteFeatureBackbone:
        """Dummy class when PyTorch is not available."""
        def __init__(self):
            raise ImportError("PyTorch required for deep feature extraction")
    
    def extract_spectral_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from multi-band image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Feature array
        """
        if image.ndim == 2:
            # Convert to 3D for consistency
            image = np.expand_dims(image, axis=-1)
        
        h, w, c = image.shape
        
        # Reshape for processing
        pixels = image.reshape(-1, c)
        
        # Calculate spectral indices if multi-band
        features = []
        
        if c >= 3:  # Assume RGB or multispectral
            # Vegetation indices (if applicable)
            red_band = pixels[:, 0]
            green_band = pixels[:, 1] if c > 1 else pixels[:, 0]
            blue_band = pixels[:, 2] if c > 2 else pixels[:, 0]
            
            # NDVI-like index (normalized difference)
            if c > 1:
                ndvi = (red_band - green_band) / (red_band + green_band + 1e-8)
                features.append(ndvi.reshape(h, w, 1))
            
            # Color ratios
            if c >= 3:
                ratio_rg = red_band / (green_band + 1e-8)
                ratio_rb = red_band / (blue_band + 1e-8)
                ratio_gb = green_band / (blue_band + 1e-8)
                
                features.extend([
                    ratio_rg.reshape(h, w, 1),
                    ratio_rb.reshape(h, w, 1),
                    ratio_gb.reshape(h, w, 1)
                ])
        
        # Add original bands
        features.append(image)
        
        # Concatenate all features
        feature_stack = np.concatenate(features, axis=-1)
        
        self.logger.info(f"Extracted spectral features: {feature_stack.shape}")
        return feature_stack
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features using GLCM and other texture measures.
        
        Args:
            image: Input image
            
        Returns:
            Texture feature array
        """
        if image.ndim == 3:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        h, w = gray.shape
        features = []
        
        # Local Binary Pattern (simplified)
        lbp = np.zeros_like(gray, dtype=np.float32)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        features.append(lbp / 255.0)
        
        # Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2) / 255.0
        features.append(gradient_mag)
        
        # Standard deviation in local window
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(local_sq_mean - local_mean**2) / 255.0
        features.append(local_std)
        
        # Stack features
        texture_features = np.stack(features, axis=-1)
        
        self.logger.info(f"Extracted texture features: {texture_features.shape}")
        return texture_features
    
    def extract_morphological_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract morphological features.
        
        Args:
            image: Input image
            
        Returns:
            Morphological feature array
        """
        if image.ndim == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        features = []
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Opening (erosion followed by dilation)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        features.append(opening / 255.0)
        
        # Closing (dilation followed by erosion)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        features.append(closing / 255.0)
        
        # Top-hat (original - opening)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        features.append(tophat / 255.0)
        
        # Black-hat (closing - original)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        features.append(blackhat / 255.0)
        
        morph_features = np.stack(features, axis=-1)
        
        self.logger.info(f"Extracted morphological features: {morph_features.shape}")
        return morph_features
    
    def perform_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Perform unsupervised clustering for pseudo-labeling.
        
        Args:
            features: Feature array (H, W, C)
            
        Returns:
            Tuple of (cluster_labels, cluster_info)
        """
        h, w, c = features.shape
        feature_vectors = features.reshape(-1, c)
        
        # Apply PCA for dimensionality reduction if needed
        if c > 10:
            pca = PCA(n_components=10)
            feature_vectors = pca.fit_transform(feature_vectors)
            self.logger.info(f"Applied PCA: {c} -> {feature_vectors.shape[1]} dimensions")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_vectors)
        
        # Reshape back to image dimensions
        cluster_map = cluster_labels.reshape(h, w)
        
        # Calculate cluster statistics
        cluster_info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'cluster_sizes': np.bincount(cluster_labels)
        }
        
        # Assign semantic labels based on cluster characteristics
        semantic_labels = self._assign_semantic_labels(cluster_info, features)
        
        self.logger.info(f"Clustering completed: {self.n_clusters} clusters")
        
        return cluster_map, {**cluster_info, 'semantic_labels': semantic_labels}
    
    def _assign_semantic_labels(self, cluster_info: Dict, features: np.ndarray) -> Dict:
        """
        Assign semantic meaning to clusters based on characteristics.
        
        Args:
            cluster_info: Cluster information
            features: Original features
            
        Returns:
            Dictionary mapping cluster IDs to semantic labels
        """
        semantic_labels = {}
        
        # Simple heuristic-based labeling
        # This would be more sophisticated in a real implementation
        land_cover_types = ['urban', 'forest', 'water', 'agriculture', 'bare_soil']
        
        for i in range(self.n_clusters):
            if i < len(land_cover_types):
                semantic_labels[i] = land_cover_types[i]
            else:
                semantic_labels[i] = f'class_{i}'
        
        return semantic_labels
    
    def extract_features(self, processed_data: Dict) -> Dict:
        """
        Main feature extraction pipeline.
        
        Args:
            processed_data: Dictionary containing preprocessed image and metadata
            
        Returns:
            Dictionary containing extracted features and analysis results
        """
        image = processed_data['image']
        
        self.logger.info("Starting feature extraction...")
        
        # Extract different types of features
        spectral_features = self.extract_spectral_features(image)
        texture_features = self.extract_texture_features(image)
        morph_features = self.extract_morphological_features(image)
        
        # Combine all features
        all_features = np.concatenate([
            spectral_features,
            texture_features,
            morph_features
        ], axis=-1)
        
        # Perform clustering for land cover classification
        cluster_map, cluster_info = self.perform_clustering(all_features)
        
        result = {
            'spectral_features': spectral_features,
            'texture_features': texture_features,
            'morphological_features': morph_features,
            'combined_features': all_features,
            'cluster_map': cluster_map,
            'cluster_info': cluster_info,
            'feature_extraction_config': self.config
        }
        
        self.logger.info("Feature extraction completed successfully")
        return result
