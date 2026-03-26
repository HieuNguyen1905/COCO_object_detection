# Object Detection with Faster R-CNN

Dự án object detection sử dụng Faster R-CNN với ResNet50 backbone để phát hiện các đối tượng trong ảnh trên COCO dataset.

## Tính năng

- **Model**: Faster R-CNN với ResNet50-FPN backbone (pretrained trên ImageNet)
- **Dataset**: COCO 2017 (118K training images, 5K validation images, 80 object classes)
- **Data Augmentation**: Albumentations với các phép biến đổi đa dạng
- **Training**: Hỗ trợ CUDA với mixed precision (nếu có)
- **Inference**: Pipeline dự đoán với visualization
- **Checkpointing**: Tự động lưu checkpoint sau mỗi epoch

## Cấu trúc dự án

```
object_detection/
├── configs/
│   └── configs.yaml              # File cấu hình chính
├── data/
│   ├── raw/
│   │   ├── train2017/           # COCO training images
│   │   └── val2017/             # COCO validation images
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
├── src/
│   ├── datasets/
│   │   ├── dataset.py           # COCO dataset loader
│   │   ├── coco_dataset.py      # COCO utilities
│   │   └── datamodule.py        # DataModule wrapper
│   ├── models/
│   │   └── bbox_regressor.py    # Faster R-CNN model builder
│   ├── training/
│   │   └── trainer.py           # Training logic
│   ├── inference/
│   │   └── predictor.py         # Inference pipeline
│   ├── pipelines/
│   │   ├── trains.py            # Training script
│   │   └── inference.py         # Inference script
│   └── utils/
│       ├── config.py            # Config loader
│       └── visualization.py     # Visualization utilities
├── outputs/
│   ├── checkpoint_epoch_*.pth   # Saved checkpoints
│   └── plots/                   # Training plots
└── README.md
```

## Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.0+ (khuyến nghị cho training)
- 16GB RAM (tối thiểu)
- GPU với 8GB+ VRAM (khuyến nghị)

## Cài đặt

### 1. Clone repository

```bash
git clone http://gitlab.technica.vn/hieunp/object_detection.git
cd object_detection
```

### 2. Tạo virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations pycocotools pyyaml tqdm opencv-python matplotlib
```

## Chuẩn bị dữ liệu

### Download COCO 2017 dataset

```bash
# Tạo thư mục data
mkdir -p data/raw data/annotations

# Download training images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/raw/

# Download validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/raw/

# Download annotations (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/

# Di chuyển annotations vào thư mục đúng
mv data/annotations/*.json data/annotations/
```

Cấu trúc thư mục data sau khi download:

```
data/
├── raw/
│   ├── train2017/          # 118,287 images
│   └── val2017/            # 5,000 images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Cấu hình

Chỉnh sửa file `configs/configs.yaml` để thay đổi hyperparameters:

```yaml
# Đường dẫn dữ liệu
TRAIN_IMAGES: "data/raw/train2017"
VAL_IMAGES: "data/raw/val2017"
TRAIN_JSON: "data/annotations/instances_train2017.json"
VAL_JSON: "data/annotations/instances_val2017.json"

# Output
BASE_OUTPUT: "outputs"
MODEL_OUTPUT: "outputs/model.pth"
PLOTS_PATH: "outputs/plots"

# Normalization (ImageNet stats)
MEAN: [0.485, 0.456, 0.406]
STD: [0.229, 0.224, 0.225]

# Model
BASE_MODEL: "resnet50"
IMAGE_SIZE: 640

# Training hyperparameters
LEARNING_RATE: 0.0001
NUM_EPOCHS: 20
BATCH_SIZE: 2              # Tăng nếu có GPU mạnh
NUM_WORKERS: 0             # Tăng để load data nhanh hơn
PIN_MEMORY: true
DEVICE: "cuda"             # "cpu" nếu không có GPU

# Loss weights
LABELS: 1.0
BBOX: 1.0
```

### Các tham số quan trọng

- **BATCH_SIZE**: 2-4 cho GPU 8GB, 8-16 cho GPU 24GB+
- **NUM_WORKERS**: 0 (single-process), 4-8 (multi-process loading)
- **IMAGE_SIZE**: 640 (default), có thể tăng lên 800-1024 cho độ chính xác cao hơn
- **LEARNING_RATE**: 0.0001 (default), có thể dùng learning rate scheduler

## Training

### Chạy training

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Start training
python3 -m src.pipelines.trains
```

### Monitor training

Training sẽ hiển thị:
- Progress bar với loss cho mỗi batch
- Training loss và validation loss sau mỗi epoch
- Checkpoints được lưu tự động vào `outputs/checkpoint_epoch_*.pth`

```
loading annotations into memory...
Done (t=4.19s)
creating index...
index created!
Epoch 1 | Step 0 | Loss: 4.7106: 100%|████████| 58633/58633 [5:23:45<00:00]
Epoch 1:
Train Loss: 2.3456
Val Loss: 1.8734
```

### Resume training từ checkpoint

Chỉnh sửa `src/pipelines/trains.py`:

```python
# Load checkpoint
checkpoint = torch.load("outputs/checkpoint_epoch_5.pth")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

## Inference

### Chạy inference trên ảnh

```bash
python3 -m src.pipelines.inference --image path/to/image.jpg --checkpoint outputs/checkpoint_epoch_20.pth
```

### Sử dụng trong code

```python
from src.models.bbox_regressor import build_model
from src.inference.predictor import Predictor
import torch

# Load model
model = build_model(num_classes=91, pretrained=False)
checkpoint = torch.load("outputs/checkpoint_epoch_20.pth")
model.load_state_dict(checkpoint["model"])

# Create predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = Predictor(model, device)

# Predict
predictions = predictor.predict("path/to/image.jpg")
```

## Data Augmentation

Training sử dụng các phép augmentation sau (Albumentations):

- **Geometric transforms**:
  - Resize với LongestMaxSize: 640px
  - Padding để đảm bảo kích thước cố định
  - Horizontal flip (p=0.5)

- **Color transforms**:
  - ColorJitter: brightness, contrast, saturation, hue (p=0.5)
  - RandomBrightnessContrast (p=0.3)

- **Noise/Blur**:
  - GaussianBlur (p=0.2)

Validation chỉ sử dụng resize + padding (không augmentation).

## Model Architecture

### Faster R-CNN Components

1. **Backbone**: ResNet50 với Feature Pyramid Network (FPN)
   - Pretrained trên ImageNet
   - Tạo multi-scale feature maps

2. **Region Proposal Network (RPN)**
   - Đề xuất các vùng có khả năng chứa object
   - Anchor boxes với nhiều scales và aspect ratios

3. **ROI Pooling & Heads**
   - Box predictor: dự đoán class và bounding box refinement
   - Fast R-CNN predictor với 91 classes (80 COCO + background)

### Loss Function

Faster R-CNN sử dụng multi-task loss:
- **Classification loss**: Cross-entropy cho object classes
- **Bounding box regression loss**: Smooth L1 loss cho box coordinates
- **RPN losses**: Classification và regression cho region proposals

## Performance Tips

### Tăng tốc training

1. **Tăng batch size**: Nếu GPU có đủ memory
   ```yaml
   BATCH_SIZE: 8  # thay vì 2
   ```

2. **Tăng num_workers**: Parallel data loading
   ```yaml
   NUM_WORKERS: 4  # hoặc 8
   ```

3. **Mixed precision training**: Thêm vào trainer
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

4. **Reduce image size**: Nếu cần training nhanh hơn
   ```yaml
   IMAGE_SIZE: 512  # thay vì 640
   ```

### GPU Memory optimization

- Giảm BATCH_SIZE nếu gặp CUDA out of memory
- Giảm IMAGE_SIZE
- Sử dụng gradient accumulation cho effective larger batch size

## Troubleshooting

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Giải pháp**: Giảm `BATCH_SIZE` hoặc `IMAGE_SIZE` trong configs

### Slow data loading
```
Training is slow, bottleneck at data loading
```
**Giải pháp**: Tăng `NUM_WORKERS` trong configs (4-8)

### Annotation errors
```
KeyError: 'boxes' or 'labels'
```
**Giải pháp**: Kiểm tra format của COCO annotations và đường dẫn trong configs

## COCO Classes

Model được train với 80 object classes của COCO dataset:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

## References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Albumentations](https://albumentations.ai/)

## License

[Chọn license phù hợp: MIT, Apache 2.0, GPL, etc.]

## Contributing

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## Author

Hieu Nguyen

---

**Note**: Dự án này được phát triển cho mục đích học tập và nghiên cứu.
