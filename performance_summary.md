# DeepArUco Performance Summary

## Overview
This document summarizes the performance characteristics of the DeepArUco marker detection system.

## Test Environment
- Hardware: AMD Ryzen 9 7940HS
- OS: Linux 5.17.9-xanmod1
- Python: 3.8
- CUDA: Not available (CPU only)

## Performance Metrics

### Overall Performance
- First run (warmup): 1175.2 ms
- Mean execution time: 22.2 ms
- Standard deviation: 1.6 ms

### Detailed Step Statistics
| Step | Mean (ms) | Std (ms) |
|------|-----------|----------|
| Detection | 10.1 | 1.0 |
| BBox Expansion | 0.1 | 0.0 |
| Crop Normalize | 0.1 | 0.0 |
| Corner Refinement | 3.2 | 0.5 |
| Corner Conversion | 0.7 | 0.1 |
| Corner Filtering | 0.0 | 0.0 |
| Corner Ordering | 0.1 | 0.0 |
| Marker Extraction | 0.1 | 0.0 |
| Marker Decoding | 7.6 | 0.8 |

## Notes
- Performance measurements exclude the first run (warmup)
- All timings are in milliseconds
- The system achieves real-time performance with consistent timing
- The most time-consuming steps are:
  1. Detection (10.1ms)
  2. Marker decoding (7.6ms)
  3. Corner refinement (3.2ms) 