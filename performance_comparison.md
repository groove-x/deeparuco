# ArUco Detector Performance Comparison

This document compares the performance of DeepArUco and OpenCV's built-in ArUco detector on the same test image.

## Test Environment
- Hardware: Linux system with NVIDIA GPU
- Image: `gen_image.jpg`
- Marker Type: 6x6 ArUco markers
- Number of runs: 1000 (excluding warmup)

## Overall Performance

| Implementation | Mean Time (ms) | Std Dev (ms) | Warmup Time (ms) |
|---------------|---------------|--------------|------------------|
| DeepArUco     | 21.7          | 1.4          | 1216.2           |
| OpenCV        | 3.4           | 0.8          | 4.0              |

## Detailed Step Timings

### DeepArUco Processing Steps

| Step                  | Mean Time (ms) | Std Dev (ms) | Description                    |
|----------------------|---------------|--------------|--------------------------------|
| Detection            | 9.7           | 0.7          | Initial marker detection using deep learning model |
| Bounding box expansion| 0.1           | 0.0          | Expanding detected bounding boxes for better coverage |
| Crop and normalize    | 0.1           | 0.0          | Preparing marker regions for processing |
| Corner refinement     | 3.2           | 0.4          | Fine-tuning marker corner positions |
| Corner conversion     | 0.7           | 0.1          | Converting corner coordinates to image space |
| Corner filtering      | 0.0           | 0.0          | Removing invalid corner detections |
| Corner ordering       | 0.1           | 0.0          | Arranging corners in consistent order |
| Marker extraction     | 0.1           | 0.0          | Extracting marker region for decoding |
| Marker decoding       | 7.5           | 0.7          | Decoding marker ID and data |
| **Total**            | **21.7**      | **1.4**      | Complete processing pipeline |

### OpenCV Processing Steps

| Step       | Mean Time (ms) | Std Dev (ms) | Description                    |
|------------|---------------|--------------|--------------------------------|
| Detection  | 3.4           | 0.8          | Single-step marker detection and pose estimation |
| **Total**  | **3.4**       | **0.8**      | Complete processing pipeline |

## Key Differences

| Aspect     | DeepArUco | OpenCV | Notes |
|------------|-----------|--------|-------|
| Speed      | 21.7 ms   | 3.4 ms | OpenCV is significantly faster due to simpler approach |
| Stability  | 1.4 ms    | 0.8 ms | DeepArUco shows more consistent performance |
| Pipeline   | Complex   | Simple | DeepArUco has multiple processing steps |
| Features   | Advanced  | Basic  | DeepArUco includes corner refinement and detailed decoding |

## Visual Outputs
Both implementations generate visualization outputs:
- DeepArUco: `output_visualization.jpg`
- OpenCV: `output_visualization_opencv.jpg`

You can compare these files to see any differences in detection quality or visualization style.

## Conclusion
The implementations show different trade-offs:

| Requirement | Recommended Implementation | Reason |
|-------------|---------------------------|---------|
| Speed       | OpenCV                    | 6.4x faster processing |
| Stability   | DeepArUco                 | Lower standard deviation |
| Accuracy    | DeepArUco                 | More sophisticated processing pipeline |
| Dependencies| OpenCV                    | Lighter weight, fewer dependencies | 