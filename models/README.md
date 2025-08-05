# Model Weights Directory

This directory contains the AI model weights and configurations for the GIS Image Analysis project.

## Available Models

### YOLOv8 Object Detection
- **Models**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **Purpose**: Object detection in aerial/satellite imagery
- **Classes**: Buildings, vehicles, roads, bridges, aircraft, ships, etc.
- **Weights**: Automatically downloaded from Ultralytics
- **Location**: ~/.ultralytics/models/ (managed by ultralytics)

### DeepLabV3+ Semantic Segmentation  
- **Architecture**: DeepLabV3+ with ResNet101 encoder
- **Purpose**: Land use classification
- **Classes**: Background, vegetation, urban, water, agriculture, bare soil
- **Weights**: ImageNet pretrained encoder
- **Location**: ~/.cache/torch/hub/ (managed by PyTorch)

## Custom Model Training

To use custom trained models:

1. **YOLOv8**: Place your trained weights as `yolov8{size}_aerial.pt`
2. **DeepLabV3+**: Place your trained weights as `deeplabv3plus_landuse.pt`

## Model Performance

- **YOLOv8x**: Highest accuracy, slowest inference (~2-5s per image)
- **YOLOv8l**: Good balance of speed and accuracy (~1-3s per image)  
- **YOLOv8m**: Moderate accuracy, faster inference (~0.5-1s per image)
- **YOLOv8s**: Lower accuracy, fast inference (~0.2-0.5s per image)
- **YOLOv8n**: Lowest accuracy, fastest inference (~0.1-0.3s per image)

The default configuration uses YOLOv8x for maximum accuracy.

## GPU Requirements

- **Minimum VRAM**: 4GB for YOLOv8n + DeepLabV3+
- **Recommended VRAM**: 8GB+ for YOLOv8x + DeepLabV3+
- **CPU Fallback**: All models work on CPU (slower)
