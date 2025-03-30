# YOLOX Replacement Plan

## Current Implementation Analysis

The DeepArUco++ library currently uses YOLOv8 from Ultralytics for marker detection in two main files:

1. `train_detector.py`: Used for training the marker detection model
2. `demo.py`: Used for inference/detection of markers in images

The current workflow is:
1. Dataset is built using `build_dataset.py` and processed using `build_detection.py`
2. Training is done using YOLOv8 with a custom dataset format
3. Inference uses the trained YOLOv8 model to detect marker bounding boxes

## Required Changes

### 1. Dependencies
- Remove `ultralytics` package dependency
- Add YOLOX package dependency (likely from Megvii's implementation)

### 2. Training Script (`train_detector.py`)
- Replace YOLOv8 model initialization with YOLOX
- Update training configuration to match YOLOX's requirements
- Ensure dataset format compatibility with YOLOX
- Update model saving/loading logic

### 3. Inference Script (`demo.py`)
- Replace YOLOv8 model loading with YOLOX
- Update inference code to handle YOLOX's output format
- Ensure bounding box format compatibility

### 4. Dataset Format
- Verify if current dataset format (YOLO format) is compatible with YOLOX
- If not, create conversion script or update dataset generation

## Implementation Steps

1. Create new branch `feature/yolox_replacement`
2. Update dependencies in requirements.txt
3. Modify `train_detector.py`:
   - Replace YOLOv8 imports with YOLOX
   - Update model initialization and training code
   - Test training pipeline
4. Modify `demo.py`:
   - Replace YOLOv8 imports with YOLOX
   - Update inference code
   - Test inference pipeline
5. Add tests to verify functionality
6. Update documentation

## Potential Challenges

1. Dataset Format Compatibility:
   - YOLOX may have different dataset format requirements
   - May need to modify `build_detection.py`

2. Model Architecture Differences:
   - YOLOX has different model variants
   - Need to choose appropriate model size/configuration

3. Training Parameters:
   - YOLOX may have different hyperparameters
   - Need to tune for optimal performance

4. Output Format:
   - YOLOX may output bounding boxes in different format
   - Need to ensure compatibility with corner refinement step

## Success Criteria

1. Training pipeline works with YOLOX
2. Inference produces same format output as current implementation
3. Detection accuracy is comparable or better than YOLOv8
4. All tests pass
5. Documentation is updated

## Timeline

1. Branch creation and dependency updates: 1 day
2. Training script modification: 2-3 days
3. Inference script modification: 1-2 days
4. Testing and validation: 2-3 days
5. Documentation updates: 1 day

Total estimated time: 7-10 days

## Questions for Review

1. Should we maintain backward compatibility with existing trained models?
no, we don't need to support the existing trained models or yolov8. please remove them.
2. Which YOLOX model variant should we use (nano, tiny, s, m, l, x)?
let's try with "s"
3. Should we implement any additional features available in YOLOX?
let's start with mimimal implementation first.
4. Do we need to modify the dataset generation pipeline? 
if possible, I'd like to use the generation pipeline as is without any modification.
if needed, let's discuss how to modify.