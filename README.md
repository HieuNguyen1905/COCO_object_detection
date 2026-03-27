Markdown
# Object Detection with Faster R-CNN

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-ready object detection framework using **Faster R-CNN with ResNet50-FPN** backbone trained on the **COCO 2017** dataset. This project demonstrates best practices in deep learning workflows including modular architecture, configuration management, and robust training/inference pipelines.

## ✨ Features

- **State-of-the-art Model**: Faster R-CNN with ResNet50-FPN backbone, pretrained on ImageNet
- **COCO Dataset Support**: Full support for COCO 2017 format (80 object classes)
- **Advanced Augmentation**: Albumentations integration with automatic bounding box transformation
- **Configuration-Driven**: Centralized YAML-based hyperparameter management
- **Modular Architecture**: Clean separation of concerns (datasets, models, training, inference)
- **Production-Ready**: Checkpoint management, evaluation metrics (mAP), and flexible inference pipeline
- **Modern Dependencies**: Python 3.12+, PyTorch 2.1+, with optional GPU acceleration

## 📂 Project Structure

```
object_detection/
├── configs/
│   └── configs.yaml              # Central configuration file (hyperparameters, paths)
├── data/
│   ├── annotations/              # COCO annotation JSON files
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   ├── mini_instances_train2017.json (for quick testing)
│   │   └── mini_instances_val2017.json
│   └── raw/
│       ├── train2017/            # Training images
│       ├── val2017/              # Validation images
│       ├── mini_train2017/       # 100 mini training images
│       └── mini_val2017/         # 50 mini validation images
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis notebook
├── outputs/
│   ├── best.pth                  # Best model checkpoint (highest mAP)
│   ├── last.pth                  # Latest model checkpoint
│   ├── inference/                # Inference output directory
├── src/
│   ├── datasets/
│   │   ├── dataset.py            # COCODetectionDataset class
│   │   └── datamodule.py         # DataModule wrapper
│   ├── models/
│   │   └── model.py              # Faster R-CNN builder
│   ├── training/
│   │   ├── trainer.py            # Training loop with mAP evaluation
│   │   ├── optimizer.py          # Optimizer configurations
│   │   └── scheduler.py          # Learning rate schedulers
│   ├── inference/
│   │   └── predictor.py          # Inference pipeline
│   ├── pipelines/
│   │   ├── train.py              # Training entry point
│   │   └── inference.py          # Inference CLI
│   └── utils/
│       ├── config.py             # Config loader
│       ├── transform.py          # Data augmentation pipelines
│       ├── metrics.py            # mAP calculation
│       └── visualization.py      # Bounding box visualization
├── tools/
│   ├── extract_mini_coco.py      # Generate mini dataset for testing
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Dependency lock file (deterministic builds)
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB RAM minimum (8GB GPU VRAM recommended)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/object_detection.git
cd object_detection
```

**2. Create virtual environment and install dependencies**

Using pip (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or using uv (faster):
```bash
# Install uv first: https://docs.astral.sh/uv/
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Dataset Setup

**Option 1: Full COCO 2017 Dataset**

```bash
# Download training images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/raw/

# Download validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/raw/

# Download annotations (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/
```

**Option 2: Mini Dataset (for quick testing)**

```bash
# Create mini COCO dataset (100 train, 50 val images)
python tools/extract_mini_coco.py

# Update configs/configs.yaml to use mini dataset:
# TRAIN_IMAGES: "data/raw/mini_train2017"
# VAL_IMAGES: "data/raw/mini_val2017"
# TRAIN_JSON: "data/annotations/mini_instances_train2017.json"
# VAL_JSON: "data/annotations/mini_instances_val2017.json"
```

## ⚙️ Configuration

Edit `configs/configs.yaml` to customize training:

```yaml
# Dataset paths
BASE_PATH: "data"
TRAIN_IMAGES: "data/raw/train2017"
VAL_IMAGES: "data/raw/val2017"
TRAIN_JSON: "data/annotations/instances_train2017.json"
VAL_JSON: "data/annotations/instances_val2017.json"
BASE_OUTPUT: "outputs"

# Model configuration
BASE_MODEL: "fasterrcnn_resnet50_fpn"
NUM_CLASSES: 81                     # 80 COCO classes + background
PRETRAINED: true
IMAGE_SIZE: 640

# Training hyperparameters
LEARNING_RATE: 0.0001
NUM_EPOCHS: 20
BATCH_SIZE: 4                       # Reduce if VRAM < 8GB
NUM_WORKERS: 2
PIN_MEMORY: false
DEVICE: "cuda"                      # or "cpu"

# Optimizer & Scheduler
OPTIMIZER: "adam"                   # adam or sgd
MOMENTUM: 0.9
SCHEDULER: "step"                   # step or cosine
STEP_SIZE: 3
GAMMA: 0.1

# Evaluation
IOU_THRESHOLD: 0.5
AP_INTERPOLATION_POINTS: 11

# Inference
CONF_THRESHOLD: 0.5
NMS_THRESHOLD: 0.5
```

### Key Configuration Tips

| Parameter | Recommendation |
|-----------|-----------------|
| `BATCH_SIZE` | 2-4 for GPU 8GB, 8-16 for 24GB+ |
| `NUM_WORKERS` | 0 on Windows/macOS, 2-4 on Linux |
| `IMAGE_SIZE` | 640 (default), 800+ for higher accuracy |
| `DEVICE` | "cuda" for GPU, "cpu" for CPU |

## 🏋️ Training

Start training with:

```bash
python -m src.pipelines.train
```

**Training Output:**

```
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
Epoch 1 | Step 10 | Loss: 1.8445:  56%|█████████████████████████████████████████████▉                                    | 14/25 [02:27<01:54, 10.43s/it][ WARN:0@148.322] global loadsave.cpp:278 findDecoder imread_('data/raw/mini_train2017/000000147328.jpg'): can't open/read file: check file path/integrity
Skipping missing/corrupted image: data/raw/mini_train2017/000000147328.jpg
Epoch 1 | Step 20 | Loss: 1.9026: 100%|██████████████████████████████████████████████████████████████████████████████████| 25/25 [04:24<00:00, 10.60s/it]

Validating...
Val Loss: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:51<00:00,  3.97s/it]
Evaluating mAP...
Inference: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:53<00:00,  4.14s/it]
Calculating mAP metrics...

Epoch 1:
Train Loss: 1.2179
Val Loss: 1.0950
mAP: 0.1851
Best model saved with mAP: 0.1851
```

**Checkpoints:**
- `outputs/best.pth` - Best model (auto-saved when mAP improves)
- `outputs/last.pth` - Latest model (saved each epoch)

### Resume Training

```python
import torch
from src.models import build_model

checkpoint = torch.load("outputs/last.pth")
model = build_model(num_classes=81)
model.load_state_dict(checkpoint["model_state_dict"])

# Continue training from epoch N+1
```

## 🔍 Inference

**Single Image:**

```bash
python -m src.pipelines.inference --image path/to/image.jpg
```

**Multiple Images:**

```bash
python -m src.pipelines.inference --image img1.jpg img2.jpg img3.jpg
```

**Image Folder:**

```bash
python -m src.pipelines.inference --image data/raw/val2017/ --output results/
```

**Custom Thresholds:**

```bash
python -m src.pipelines.inference \
    --image photo.jpg \
    --conf-thresh 0.3 \
    --nms-thresh 0.4
```

**Programmatic Usage:**

```python
from src.inference.predictor import Predictor

predictor = Predictor(
    checkpoint_path="outputs/best.pth",
    num_classes=81,
    device="cuda",
    conf_threshold=0.5,
    nms_threshold=0.5,
)

result = predictor.predict("image.jpg")
print(f"Detected: {len(result['boxes'])} objects")
for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
    print(f"Class {label}: {score:.3f} at {box}")
```

## 📊 Data Augmentation

**Training Transforms** (from `src/utils/transform.py`):
- LongestMaxSize (max 640px)
- RandomPadding
- HorizontalFlip (50%)
- ColorJitter (brightness, contrast, saturation)
- RandomBrightnessContrast (30%)
- GaussianBlur (30%)
- Normalize + ToTensor

**Validation/Inference Transforms**:
- LongestMaxSize (max 640px)
- Padding
- Normalize + ToTensor

## 🏗️ Model Architecture

### Faster R-CNN Components

```
Input Image (640×640)
        ↓
   ResNet50 Backbone (pretrained on ImageNet)
        ↓
   FPN (Feature Pyramid Network)
        ↓
   RPN (Region Proposal Network) → Anchor generation & filtering
        ↓
   ROI Align → Uniform feature extraction
        ↓
   ROI Head (Classification + Bounding Box Regression)
        ↓
   Output: Bounding boxes + class probabilities
```

### Loss Function

```
Total Loss = RPN_classification_loss + RPN_regression_loss 
           + ROI_classification_loss + ROI_regression_loss
```

## 📈 Evaluation Metrics

**mAP (mean Average Precision)**

- Calculated using COCO evaluation protocol
- IoU threshold: 0.5 (IoU@.5)
- 11-point interpolation
- Evaluated on 80 COCO object classes

```python
# Evaluation is automatic during validation
# Check training logs for mAP progression
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` (4→2) or `IMAGE_SIZE` (640→512) |
| Slow data loading | Increase `NUM_WORKERS` (0→4) or use SSD storage |
| Model not learning | Check learning rate defaults, verify data augmentation |
| 0 mAP at epoch 1 | Normal! Model needs 5-10 epochs to learn |

## 📚 COCO Object Classes (80 classes)

Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## 🎯 Performance Tips

### Training Speedup

```yaml
# GPU Configuration
DEVICE: "cuda"              # Use GPU
BATCH_SIZE: 8               # Increase if VRAM allows
NUM_WORKERS: 4              # Parallel data loading
PIN_MEMORY: true            # Faster GPU transfer
```

### Accuracy Improvement

- Increase `IMAGE_SIZE` (640 → 800-1024)
- Extend training (`NUM_EPOCHS` 20 → 50)
- Use learning rate scheduler
- Train on full COCO dataset (not mini)
- Add more data augmentation

## 🔗 References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497) - Ren et al., 2015
- [COCO Dataset](https://cocodataset.org/) - Lin et al., 2014
- [PyTorch Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Albumentations Documentation](https://albumentations.ai/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Hieu Nguyen** - [GitHub](https://github.com/HieuNguyen1905) | [Email](hieu.nguyenphuc1905@gmail.com)

---

**Note**: This project is developed for educational and research purposes to demonstrate best practices in object detection using PyTorch and modern deep learning workflows.