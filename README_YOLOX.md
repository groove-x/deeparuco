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

## Build dataset

```bash
python filter_backgrounds.py <source MSCOCO train2017 path> <filtered MSCOCO path>
python build_dataset.py <filtered MSCOCO path> <target flyingarucov2 path> -l -r
python convert_to_coco.py <target flyingarucov2 path> <target coco path>
```

## Train YOLOX

```bash
python YOLOX/tools/train.py -f YOLOX/exps/default/yolox_tiny.py -d 1 -b 16 --fp16 -c YOLOX/YOLOX_outputs/aruco_yolox_tiny/best_ckpt.pth
```
