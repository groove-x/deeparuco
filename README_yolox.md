# YOLOX Implementation for ArUco Marker Detection

This document describes the YOLOX implementation for detecting ArUco markers in images.

## Overview

The implementation uses a YOLOX-based object detection model to detect ArUco markers in images. The model is designed to be efficient and accurate, with support for different model sizes (s, m, l, x).

## Model Architecture

The YOLOX implementation includes:

- A backbone network with multiple YOLOX blocks
- A detection head for predicting bounding boxes and class scores
- Support for different model sizes with varying channel depths and block counts
- Grid-based prediction system with 3 anchors per grid cell

## Dataset Structure

The dataset should be organized as follows:
```
data/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Each image should have a corresponding label file with the same name but `.txt` extension. Label files should contain one line per marker in the format:
```
0 <x_center> <y_center> <width> <height>
```
where all values are normalized to [0, 1].

## Training

To train the model:

```bash
python train_detector.py <source_dir> <run_name> [options]
```

Options:
- `--model`: Model size (s, m, l, x) [default: s]
- `--batch_size`: Batch size for training [default: 32]
- `--epochs`: Number of training epochs [default: 100]
- `--lr`: Learning rate [default: 0.01]
- `--device`: Device to train on (cuda/cpu) [default: cuda if available]

Example:
```bash
python train_detector.py data/flyingaruco yolox_model --model s --batch_size 32 --epochs 100
```

The training process includes:
- Training and validation phases
- Automatic saving of the best model based on validation loss
- Progress tracking with loss values for both phases

## Inference

To run inference on images:

```bash
python test_detector.py <model_dir> <image_dir> [options]
```

Options:
- `--output_dir`: Directory to save results [default: results]
- `--conf_threshold`: Confidence threshold for detections [default: 0.5]
- `--device`: Device to run inference on [default: cuda if available]

Example:
```bash
python test_detector.py yolox_model data/test/images --output_dir results
```

## Model Output

The model outputs bounding boxes for detected markers in the format:
- `x1, y1, x2, y2`: Coordinates of the bounding box
- Confidence score for each detection

## Implementation Details

### Loss Function
The YOLOX loss combines:
- Box regression loss (Smooth L1)
- Objectness loss (Binary Cross Entropy)
- Classification loss (Binary Cross Entropy)

### Data Augmentation
The dataset includes:
- Random horizontal flips
- Random brightness adjustments
- Proper image resizing and padding
- Coordinate normalization

### Model Architecture
- Initial convolution layer
- Multiple YOLOX blocks with residual connections
- Detection head with 3 anchors per grid cell
- Support for different model sizes with varying channel depths

## Current Status

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