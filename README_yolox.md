# YOLOX Detector Implementation

This document describes the implementation of the YOLOX detector for ArUco marker detection in the DeepArUco++ project.

## Overview

The YOLOX detector is implemented to provide a deep learning-based approach for detecting ArUco markers in images. The implementation includes:

1. Dataset Generation
2. Model Architecture
3. Training Pipeline
4. Inference

## Dataset Generation

The dataset is generated using the COCO dataset as a base. The process involves:

1. **Source Images**: Using COCO train2017 dataset images
2. **Marker Generation**: Adding ArUco markers to images with:
   - Random positions and rotations
   - Variable sizes (minimum 32 pixels)
   - Up to 20 markers per image
   - Various effects:
     - Luma: Background-adaptive brightness
     - Borders: Variable width marker borders
     - Reflections: Simulated ink reflections

3. **Label Generation**: YOLO format labels for each marker containing:
   - Class ID (0-249 for real markers)
   - Center coordinates (x, y)
   - Width and height

## Model Architecture

The YOLOX model is implemented with the following components:

1. **Backbone**: CSPDarknet
2. **Neck**: FPN (Feature Pyramid Network)
3. **Head**: YOLOX head with:
   - Multiple detection scales
   - Anchor-free design
   - SimOTA label assignment

## Training Pipeline

The training process includes:

1. **Data Preparation**:
   - Image resizing and normalization
   - Data augmentation:
     - Random horizontal flips
     - Random brightness adjustments
     - Random contrast adjustments

2. **Loss Functions**:
   - Classification loss
   - Box regression loss
   - Objectness loss

3. **Training Parameters**:
   - Batch size: 8
   - Learning rate: 0.001
   - Optimizer: SGD with momentum
   - Scheduler: Cosine annealing

## Implementation Details

### Key Files

- `impl/yolox_detector.py`: Main YOLOX model implementation
- `impl/yolox_dataset.py`: Dataset class for training
- `train_detector.py`: Training script
- `test_detector.py`: Inference script
- `build_dataset.py`: Dataset generation script

### Model Architecture

The YOLOX model is built with:
- Input size: 640x640
- Number of classes: 250 (0-249 for ArUco markers)
- Anchor-free detection
- SimOTA label assignment

### Training Process

1. **Data Loading**:
   ```python
   dataset = YOLOXDataset(
       img_dir='data/flyingaruco/images',
       label_dir='data/flyingaruco/labels',
       transform=transforms
   )
   ```

2. **Model Training**:
   ```python
   model = YOLOXDetector(
       num_classes=250,
       size='s'  # Small model variant
   )
   ```

3. **Loss Computation**:
   ```python
   loss = criterion(predictions, targets)
   ```

## Usage

### Training

```bash
python train_detector.py data/flyingaruco yolox_model --model s --batch_size 8 --epochs 100
```

### Inference

```bash
python test_detector.py yolox_model data/test/images --output_dir results
```

## Current Status

- [x] Dataset generation pipeline
- [x] YOLOX model architecture
- [x] Training pipeline
- [x] Inference script
- [ ] Model evaluation
- [ ] Performance optimization

## Next Steps

1. Complete the dataset generation process
2. Train the model on the generated dataset
3. Evaluate model performance
4. Optimize inference speed
5. Add support for different model variants (m, l, xl)

## Notes

- The implementation uses PyTorch
- The model is designed for real-time inference
- The dataset generation process takes several hours to complete
- The model can be extended to support different ArUco marker dictionaries 