# ArUco Detector Performance Comparison

This document compares the performance of DeepArUco and OpenCV's built-in ArUco detector on the same test image.

## Test Environment
### Hardware Specifications
- CPU: AMD Ryzen 9 5900HS
  - Cores: 8 cores, 16 threads
  - Base Clock: 3.5 GHz
  - Max Boost Clock: 4.68 GHz
- GPU: NVIDIA GPU (not utilized in these tests)
- Note: DeepArUco and OpenCV both running on CPU only

### Test Configuration
- Image: `gen_image.jpeg` (with added texture)
- Marker Type: 6x6 ArUco markers
- Number of runs: 1000 (excluding warmup)
- Python Environment: Python 3.8 with TensorFlow and OpenCV

## Overall Performance

| Implementation | Mean Time (ms) | Std Dev (ms) | Warmup Time (ms) | Hardware |
|---------------|---------------|--------------|------------------|-----------|
| DeepArUco     | 21.6          | 1.6          | 1256.0           | CPU      |
| OpenCV        | 8.6           | 1.4          | 13.6             | CPU      |

## Detailed Step Timings

### DeepArUco Processing Steps (CPU-only)

| Step                  | Mean Time (ms) | Std Dev (ms) | Description                    |
|----------------------|---------------|--------------|--------------------------------|
| Detection            | 9.6           | 0.8          | Initial marker detection using deep learning model |
| Bounding box expansion| 0.1           | 0.0          | Expanding detected bounding boxes for better coverage |
| Crop and normalize    | 0.1           | 0.0          | Preparing marker regions for processing |
| Corner refinement     | 3.2           | 0.4          | Fine-tuning marker corner positions |
| Corner conversion     | 0.7           | 0.1          | Converting corner coordinates to image space |
| Corner filtering      | 0.0           | 0.0          | Removing invalid corner detections |
| Corner ordering       | 0.1           | 0.0          | Arranging corners in consistent order |
| Marker extraction     | 0.1           | 0.0          | Extracting marker region for decoding |
| Marker decoding       | 7.5           | 0.8          | Decoding marker ID and data |
| **Total**            | **21.6**      | **1.6**      | Complete processing pipeline |

### OpenCV Processing Steps

| Step       | Mean Time (ms) | Std Dev (ms) | Description                    |
|------------|---------------|--------------|--------------------------------|
| Detection  | 8.5           | 1.4          | Single-step marker detection and pose estimation |
| **Total**  | **8.6**       | **1.4**      | Complete processing pipeline |

## Key Differences

| Aspect     | DeepArUco | OpenCV | Notes |
|------------|-----------|--------|-------|
| Speed      | 21.6 ms   | 8.6 ms | OpenCV is faster, but DeepArUco running on CPU only |
| Stability  | 1.6 ms    | 1.4 ms | Both show similar stability with textured image |
| Pipeline   | Complex   | Simple | DeepArUco has multiple processing steps |
| Features   | Advanced  | Basic  | DeepArUco includes corner refinement and detailed decoding |

## Visual Outputs
Both implementations generate visualization outputs:
- DeepArUco: `output_visualization.jpg`
- OpenCV: `output_visualization_opencv.jpg`

You can compare these files to see any differences in detection quality or visualization style.

## Conclusion

| Requirement | Recommended Implementation | Reason |
|-------------|---------------------------|---------|
| Speed       | OpenCV                    | 2.5x faster processing (note: DeepArUco on CPU) |
| Stability   | Both                      | Similar stability (1.4-1.6ms std dev) |
| Accuracy    | DeepArUco                 | More sophisticated processing pipeline |
| Dependencies| OpenCV                    | Lighter weight, fewer dependencies |

### Notes on Textured Image Performance
- Both implementations showed increased processing times with the textured image
- OpenCV's performance decreased more significantly (from 3.4ms to 8.6ms)
- DeepArUco maintained more consistent performance (from 21.7ms to 21.6ms)
- The stability gap between implementations narrowed with the textured image

### Notes on Hardware Usage
- Current DeepArUco performance is measured using CPU only (AMD Ryzen 9 5900HS)
- The two most time-consuming steps (detection: 9.6ms and marker decoding: 7.5ms) could potentially benefit significantly from GPU acceleration
- GPU acceleration could potentially bring DeepArUco's performance closer to or possibly better than OpenCV's
- The YOLO detector especially is known to perform significantly better on GPU, with typical speedups of 10-20x on modern GPUs
- CPU performance is already quite good considering the complexity of the deep learning pipeline
- Both implementations benefit from the high-performance CPU with multiple cores, though the degree of parallelization may vary 