# DeepArUco++ Performance Analysis

This document summarizes the performance analysis of DeepArUco++ marker detection on an Apple M3 device.

## Test Environment
- Device: Apple M3
- System Memory: 16.00 GB
- Python Version: 3.9.18
- TensorFlow Version: 2.9.2 (with Metal support)
- Test Image: `hornfront_0_0.12_0.95.jpeg`
- Number of Runs: 1000 (excluding warmup)

## Overall Performance
- Mean Execution Time: 89.0 ms
- Standard Deviation: 12.0 ms
- Warmup Time: 1157.0 ms

## Step-by-Step Breakdown

| Step | Mean Time (ms) | Std Dev (ms) | % of Total |
|------|---------------|--------------|------------|
| Detection | 76.0 | 12.0 | 85.4% |
| Corner Refinement | 5.0 | 1.0 | 5.6% |
| Marker Decoding | 7.0 | 1.0 | 7.9% |
| Bbox Expansion | <1.0 | <1.0 | <1.1% |
| Crop Normalize | <1.0 | <1.0 | <1.1% |
| Corner Conversion | <1.0 | <1.0 | <1.1% |
| Corner Filtering | <1.0 | <1.0 | <1.1% |
| Corner Ordering | <1.0 | <1.0 | <1.1% |
| Marker Extraction | <1.0 | <1.0 | <1.1% |

## Key Observations

1. **Bottleneck**: The YOLO detection model is the main bottleneck, accounting for 85.4% of the total processing time.

2. **Model Inference**: The three deep learning models (detection, refinement, and decoding) together account for 98.9% of the processing time:
   - YOLO Detection: 76.0 ms
   - Corner Refinement: 5.0 ms
   - Marker Decoding: 7.0 ms

3. **Pre/Post-processing**: All other operations (cropping, normalization, corner processing) are very efficient, each taking less than 1 ms.

4. **Stability**: The system shows good stability after warmup, with a standard deviation of 12.0 ms (13.5% of mean time).

5. **Warmup Impact**: The first run (warmup) is significantly slower (1157.0 ms) due to model loading and initialization.

## Performance Characteristics

- **Consistency**: Very stable performance after warmup
- **Real-time Capability**: Processing time of ~89 ms translates to approximately 11 FPS
- **Resource Usage**: Efficient memory usage with minimal overhead in pre/post-processing steps 