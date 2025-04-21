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

```python3
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define dataset path
        data_root = "<target coco path>"
        self.data_dir = f"{data_root}/images"
        self.train_ann = f"{data_root}/annotations/train_annotations.json"
        self.val_ann = f"{data_root}/annotations/valid_annotations.json"
 
        self.num_classes = 1
```

```bash
python tools/train.py -f exps/default/yolox_tiny.py
```
