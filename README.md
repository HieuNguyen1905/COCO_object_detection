# Object Detection with Faster R-CNN

Dự án object detection sử dụng Faster R-CNN với ResNet50 backbone để phát hiện các đối tượng trong ảnh trên COCO dataset.

## ✨ Tính năng

- **Model**: Faster R-CNN với ResNet50-FPN backbone (pretrained trên ImageNet)
- **Dataset**: COCO 2017 (118K training images, 5K validation images, 80 object classes)
- **Data Augmentation**: Albumentations với các phép biến đổi đa dạng
- **Training**: Hỗ trợ CUDA, automatic checkpoint management (best/last)
- **Evaluation**: Tích hợp mAP (mean Average Precision) calculation
- **Inference**: Pipeline dự đoán với visualization và flexible CLI
- **Mini Dataset**: Hỗ trợ tạo mini dataset để test code nhanh

## 📁 Cấu trúc dự án

```
object_detection/
├── configs/
│   └── configs.yaml              # File cấu hình chính
├── data/
│   ├── raw/
│   │   ├── train2017/           # COCO training images
│   │   ├── val2017/             # COCO validation images
│   │   ├── mini_train2017/      # Mini training set (100 images)
│   │   └── mini_val2017/        # Mini validation set (50 images)
│   └── annotations/
│       ├── instances_train2017.json
│       ├── instances_val2017.json
│       ├── mini_instances_train2017.json
│       └── mini_instances_val2017.json
├── src/
│   ├── datasets/
│   │   ├── dataset.py           # COCO dataset loader
│   │   └── datamodule.py        # DataModule wrapper
│   ├── models/
│   │   └── bbox_regressor.py    # Faster R-CNN model builder
│   ├── training/
│   │   ├── trainer.py           # Training logic với mAP evaluation
│   │   ├── optimizer.py         # Optimizer configurations
│   │   └── scheduler.py         # Learning rate schedulers
│   ├── inference/
│   │   └── predictor.py         # Inference pipeline
│   ├── pipelines/
│   │   ├── train.py             # Training script
│   │   └── inference.py         # Inference CLI
│   └── utils/
│       ├── config.py            # Config loader
│       ├── transform.py         # Data augmentation pipelines
│       ├── metrics.py           # mAP calculation
│       └── visualization.py     # Visualization utilities
├── tools/
│   ├── extract_mini_coco.py     # Tạo mini dataset
│   ├── debug_predictions.py     # Debug tool
│   └── test_inference.py        # Quick test inference
├── outputs/
│   ├── best.pth                # Best checkpoint (highest mAP)
│   ├── last.pth                # Latest checkpoint
│   └── inference/              # Inference results
└── README.md
```

## 📋 Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.0+ (khuyến nghị cho training)
- 16GB RAM (tối thiểu)
- GPU với 8GB+ VRAM (khuyến nghị)

## 🔧 Cài đặt

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
pip install albumentations pycocotools pyyaml tqdm opencv-python matplotlib numpy
```

## 📦 Chuẩn bị dữ liệu

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

### Tạo Mini Dataset (cho quick testing)

```bash
# Tạo mini train (100 images) và mini val (50 images)
python3 tools/extract_mini_coco.py
```

Sau khi chạy, cập nhật `configs/configs.yaml`:

```yaml
# Dùng mini dataset để test nhanh
TRAIN_IMAGES: "data/raw/mini_train2017"
VAL_IMAGES: "data/raw/mini_val2017"
TRAIN_JSON: "data/annotations/mini_instances_train2017.json"
VAL_JSON: "data/annotations/mini_instances_val2017.json"

# Hoặc dùng full dataset để train production model
# TRAIN_IMAGES: "data/raw/train2017"
# VAL_IMAGES: "data/raw/val2017"
# TRAIN_JSON: "data/annotations/instances_train2017.json"
# VAL_JSON: "data/annotations/instances_val2017.json"
```

## ⚙️ Cấu hình

File `configs/configs.yaml`:

```yaml
# Dataset paths
BASE_PATH: "data"
TRAIN_IMAGES: "data/raw/mini_train2017"
VAL_IMAGES: "data/raw/mini_val2017"
TRAIN_JSON: "data/annotations/mini_instances_train2017.json"
VAL_JSON: "data/annotations/mini_instances_val2017.json"

# Output
BASE_OUTPUT: "outputs"

# Model
BASE_MODEL: "resnet50"
IMAGE_SIZE: 640

# Training hyperparameters
LEARNING_RATE: 0.0001
NUM_EPOCHS: 20
BATCH_SIZE: 4              # Tăng nếu có GPU mạnh
NUM_WORKERS: 2             # Tăng để load data nhanh hơn
PIN_MEMORY: false
DEVICE: "cuda"             # "cpu" nếu không có GPU
```

### Các tham số quan trọng

- **BATCH_SIZE**: 2-4 cho GPU 8GB, 8-16 cho GPU 24GB+
- **NUM_WORKERS**: 0 (single-process), 2-4 (multi-process loading)
- **IMAGE_SIZE**: 640 (default), có thể tăng lên 800-1024 cho độ chính xác cao hơn

## 🚀 Training

### Chạy training

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Start training
python3 -m src.pipelines.train
```

### Output training

Training sẽ hiển thị:
- **Progress bar** với loss cho mỗi batch
- **Validation loss** và **mAP** sau mỗi epoch
- **Auto-save** best.pth (khi mAP tăng) và last.pth (mỗi epoch)

```
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!

Epoch 1 | Step 20 | Loss: 1.5283: 100%|████████| 25/25 [04:09<00:00, 10.56s/it]

Validating...
Val Loss: 100%|████████████████| 13/13 [00:51<00:00, 3.97s/it]
Evaluating mAP...
Inference: 100%|█████████████████| 13/13 [00:53<00:00, 4.14s/it]
Calculating mAP metrics...
Debug: Total predictions: 450, Total targets: 280
Score range: [0.0234, 0.9876], Mean: 0.4521
Class 1: 45 preds, 32 targets
  -> AP: 0.4123
...
Final mAP from 15 classes: 0.4094

Epoch 1:
Train Loss: 1.7660
Val Loss: 1.1084
mAP: 0.4094
Best model saved with mAP: 0.4094
```

### Checkpoint Management

- **best.pth**: Checkpoint với mAP cao nhất (tự động lưu khi mAP cải thiện)
- **last.pth**: Checkpoint mới nhất (lưu sau mỗi epoch)

### Resume training từ checkpoint

```python
# Trong src/pipelines/train.py, thêm:
import os

checkpoint_path = "outputs/last.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}, mAP: {checkpoint['mAP']:.4f}")
```

## 🔍 Inference

### 1. Single image

```bash
# Tự động dùng best.pth
python -m src.pipelines.inference --image path/to/image.jpg

# Chỉ định checkpoint
python -m src.pipelines.inference \
    --image path/to/image.jpg \
    --weights outputs/last.pth
```

### 2. Multiple images

```bash
python -m src.pipelines.inference \
    --image img1.jpg img2.jpg img3.jpg
```

### 3. Folder of images

```bash
python -m src.pipelines.inference \
    --image data/raw/mini_val2017/ \
    --output results/
```

### 4. Adjust thresholds

```bash
python -m src.pipelines.inference \
    --image photo.jpg \
    --conf-thresh 0.3 \
    --nms-thresh 0.4
```

### 5. Print only (no save)

```bash
python -m src.pipelines.inference \
    --image photo.jpg \
    --no-save
```

### Output

```
Processing 000000016228.jpg ...
  Detections: 5
              person  0.987  [145, 234, 456, 678]
                 car  0.892  [23, 45, 198, 234]
                  dog  0.765  [345, 123, 456, 345]
  Saved → outputs/inference/000000016228_det.jpg
```

### Quick test

```bash
# Test nhanh trên 3 ảnh validation
python3 tools/test_inference.py
```

### Sử dụng trong code

```python
from src.inference.predictor import Predictor

# Initialize predictor
predictor = Predictor(
    checkpoint_path="outputs/best.pth",
    num_classes=81,  # 80 classes + background
    device="cuda",
    conf_threshold=0.5,
    nms_threshold=0.5,
)

# Predict
result = predictor.predict("path/to/image.jpg")

print(f"Found {len(result['boxes'])} objects")
for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
    print(f"Class {label}: {score:.3f} at {box}")
```

## 📊 Data Augmentation

### Training transforms (src/utils/transform.py)

```python
- LongestMaxSize(max_size=640)
- PadIfNeeded(min_height=640, min_width=640)
- HorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
- RandomBrightnessContrast(p=0.3)
- GaussianBlur(blur_limit=(3, 7), p=0.2)
- ToTensorV2()
```

### Validation/Inference transforms

```python
- LongestMaxSize(max_size=640)
- PadIfNeeded(min_height=640, min_width=640)
- ToTensorV2()
```

## 🏗️ Model Architecture

### Faster R-CNN Components

1. **Backbone**: ResNet50-FPN
   - Pretrained trên ImageNet
   - Feature Pyramid Network cho multi-scale features

2. **Region Proposal Network (RPN)**
   - Đề xuất regions có khả năng chứa objects
   - Anchor-based với multiple scales/ratios

3. **ROI Head**
   - Box classifier: 81 classes (80 COCO + background)
   - Box regressor: bounding box refinement

### Loss Components

```
Total Loss = RPN_cls + RPN_reg + ROI_cls + ROI_reg
```

## 📈 Evaluation Metrics

### mAP (mean Average Precision)

- **AP per class**: Average Precision cho mỗi class
- **mAP**: Mean của tất cả class APs
- **IoU threshold**: 0.5 (default)
- **Evaluation**: 11-point interpolation

```python
# Trong src/utils/metrics.py
def evaluate_map(model, dataloader, device, num_classes, iou_threshold=0.5):
    # Inference trên validation set
    # Calculate precision-recall cho mỗi class
    # Compute AP và average thành mAP
    ...
```

## 🎯 Performance Tips

### Tăng tốc training

1. **GPU**: Sử dụng CUDA thay vì CPU
   ```yaml
   DEVICE: "cuda"
   ```

2. **Batch size**: Tăng nếu GPU có đủ memory
   ```yaml
   BATCH_SIZE: 8
   ```

3. **Num workers**: Parallel data loading
   ```yaml
   NUM_WORKERS: 4
   ```

4. **Pin memory**: Tăng tốc transfer GPU
   ```yaml
   PIN_MEMORY: true
   ```

### Tăng độ chính xác

1. **Image size lớn hơn**:
   ```yaml
   IMAGE_SIZE: 800  # thay vì 640
   ```

2. **Train lâu hơn**:
   ```yaml
   NUM_EPOCHS: 50
   ```

3. **Learning rate scheduler**: Giảm LR theo thời gian

4. **Use full dataset** thay vì mini dataset

## 🐛 Troubleshooting

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Giải pháp**:
- Giảm `BATCH_SIZE` (ví dụ: 4 → 2)
- Giảm `IMAGE_SIZE` (640 → 512)
- Set `NUM_WORKERS: 0`

### Slow data loading

```
Training bottleneck at data loading
```

**Giải pháp**:
- Tăng `NUM_WORKERS` (0 → 2 → 4)
- Set `PIN_MEMORY: true`
- Sử dụng SSD thay vì HDD

### Model không học (loss không giảm)

**Giải pháp**:
- Kiểm tra learning rate (thử 0.0001, 0.001, 0.00001)
- Kiểm tra data augmentation (tắt một số để test)
- Verify dataset format (boxes, labels đúng format)

### mAP = 0 ở epoch đầu

**Bình thường!** Model chưa học được:
- Chờ thêm 5-10 epochs
- mAP sẽ tăng dần khi model học
- Check predictions với `tools/debug_predictions.py`

## 📚 COCO Classes (80 classes)

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

## 🔗 References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497) - Ren et al., 2015
- [COCO Dataset](https://cocodataset.org/) - Lin et al., 2014
- [PyTorch Torchvision](https://pytorch.org/vision/stable/models.html)
- [Albumentations](https://albumentations.ai/) - Fast image augmentation library

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Hieu Nguyen**
- GitLab: [@hieunp](http://gitlab.technica.vn/hieunp)

---

**Note**: Dự án này được phát triển cho mục đích học tập và nghiên cứu về Object Detection với Faster R-CNN.
