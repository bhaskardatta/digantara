<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# GIS Image Analysis Project Instructions

## Project Context
This is a comprehensive GIS image analysis project focused on extracting features from high-quality TIFF aerial/satellite imagery using state-of-the-art AI/ML techniques.

## Key Technologies
- **Computer Vision**: YOLOv8 for object detection, DeepLabV3+ for semantic segmentation
- **Geospatial**: rasterio, GDAL for TIFF processing, geopandas for spatial data
- **Deep Learning**: PyTorch ecosystem with pre-trained models
- **Visualization**: matplotlib, folium, plotly for maps and overlays

## Code Style Guidelines
- Follow PEP 8 conventions
- Use type hints for all function parameters and returns
- Include comprehensive docstrings with Google style
- Implement error handling with appropriate logging
- Create modular, reusable components

## Architecture Principles
- **Modular Design**: Separate concerns (preprocessing, detection, segmentation, visualization)
- **Configuration-Driven**: Use config files for model parameters and thresholds
- **Pipeline-Based**: Chain operations for reproducible workflows
- **Performance-Optimized**: Handle large TIFF files efficiently with chunking

## Specific Considerations
- Handle multi-band TIFF images (RGB, NIR, etc.)
- Implement proper coordinate reference system (CRS) handling
- Use GPU acceleration where available
- Create memory-efficient data loaders for large images
- Generate geospatially-aware outputs with proper metadata

## Testing Strategy
- Unit tests for each module
- Integration tests for full pipeline
- Performance benchmarks for large images
- Validation against known datasets

## Documentation Requirements
- API documentation for all public methods
- Usage examples in docstrings
- Jupyter notebooks for tutorials
- Technical report with methodology and results
