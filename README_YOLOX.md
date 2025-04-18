# Setup

## Create virtual environment
```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_yolox.txt
```

## To update requirements

```bash
pip-compile requirements.in
pip-compile -o requirements_yolox.in YOLOX/requirements.txt
```

## Run demo with YOLOX model

```bash
python demo.py -f impl/yolox_tiny.py --ckpt ./YOLOX/YOLOX_outputs/aruco_yolox_tiny/best_ckpt.pth -c \
    examples/flyingaruco_1.jpg ./output.png
```
