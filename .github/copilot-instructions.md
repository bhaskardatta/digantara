<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# GIS Image Analysis Project Instructions

This is a Python project for analyzing GIS TIFF images using AI/ML techniques. The project focuses on:

1. **Preprocessing**: TIFF image loading, normalization, and noise handling
2. **Feature Extraction**: CNN-based land cover classification and object detection
3. **Model Development**: U-Net for segmentation, ResNet for classification
4. **Evaluation**: Accuracy metrics, IoU, and visualization

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings with parameter descriptions
- Follow PEP 8 styling conventions
- Use meaningful variable names related to GIS/ML terminology
- Add error handling for file I/O operations with TIFF files

## Libraries and Frameworks
- Primary ML frameworks: PyTorch, TensorFlow/Keras
- GIS libraries: rasterio, geopandas, shapely
- Image processing: opencv-python, scikit-image, Pillow
- Visualization: matplotlib, seaborn, folium, plotly

## Project-Specific Context
- Working with high-resolution TIFF images
- Land cover classification categories: urban, forest, water, agriculture
- Unsupervised learning approach with pseudo-labeling
- Output formats: PNG/TIFF visualizations and numerical results
- Modular code structure with separate preprocessing, modeling, and evaluation components
