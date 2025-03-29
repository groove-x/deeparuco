# DeepArUco++ Environment Setup Guide

This guide will help you set up the environment for running DeepArUco++ on macOS (Apple Silicon).

## Prerequisites

1. Python 3.9.18 (recommended for compatibility)
2. pyenv (for Python version management)
3. Homebrew (for system dependencies)

## Setup Steps

1. Install Python 3.9.18 with required dependencies:
```bash
PYTHON_CONFIGURE_OPTS="--enable-framework --with-lzma" pyenv install --force 3.9.18
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install tensorflow-macos==2.9.2 tensorflow-metal==0.5.1 numpy==1.24.3 opencv-python ultralytics scikit-image scikit-learn matplotlib tqdm shapely
```

## Running the Demo

1. Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

2. Run the demo with a sample image:
```bash
python demo.py examples/shadowaruco_1.png output_result.png
```

## Troubleshooting

If you encounter any issues:

1. Make sure you're using Python 3.9.18
2. Verify that all packages are installed correctly
3. Check that you're in the virtual environment
4. Ensure you have sufficient system memory (recommended: 16GB+)

## Notes

- The setup has been tested on macOS with Apple Silicon (M3)
- TensorFlow 2.9.2 is used for compatibility with the model architecture
- NumPy 1.24.3 is required for compatibility with TensorFlow 2.9.2 