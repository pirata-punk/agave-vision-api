# Training Workbench

Dedicated workspace for training YOLO models with version management and evaluation.

## Overview

The training module provides tools for training, evaluating, and managing YOLO model versions.

## Quick Start

### Train a New Model

```bash
python training/cli.py train \
    --data-yaml configs/yolo_data.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch 16 \
    --device cuda \
    --version v1_baseline
```

**Output:**
- `training/runs/v1_baseline/weights/best.pt` - Best model weights
- `training/runs/v1_baseline/weights/last.pt` - Latest checkpoint
- `training/runs/v1_baseline/results.csv` - Training metrics
- `training/runs/v1_baseline/training_config.json` - Configuration

### Resume Training

```bash
python training/cli.py resume \
    --checkpoint training/runs/v1_baseline/weights/last.pt
```

### Evaluate Model

```bash
python training/cli.py evaluate \
    --model training/runs/v1_baseline/weights/best.pt \
    --data-yaml configs/yolo_data.yaml \
    --split test
```

**Metrics:**
- mAP@0.5 - Mean Average Precision at IoU 0.5
- mAP@0.5-0.95 - Mean Average Precision across IoU thresholds
- Precision - True Positives / (True Positives + False Positives)
- Recall - True Positives / (True Positives + False Negatives)

### Compare Models

```bash
python training/cli.py compare \
    training/runs/v1_baseline/weights/best.pt \
    training/runs/v2_improved/weights/best.pt \
    training/runs/v3_optimized/weights/best.pt \
    --split test
```

## Programmatic Usage

```python
from agave_vision.training.trainer import YOLOTrainer
from agave_vision.training.evaluator import ModelEvaluator

# Train model
trainer = YOLOTrainer(
    data_yaml="configs/yolo_data.yaml",
    model_name="yolov8n",
    version_name="v1_baseline"
)

results = trainer.train(
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda"
)

# Evaluate model
evaluator = ModelEvaluator(
    model_path="training/runs/v1_baseline/weights/best.pt",
    data_yaml="configs/yolo_data.yaml"
)

metrics = evaluator.evaluate(split="test")
print(f"mAP@0.5: {metrics['metrics']['map50']:.4f}")
```

## Training Configuration

### Hyperparameters

Create `training/configs/hyperparameters.yaml`:

```yaml
# Model architecture
model: yolov8n  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Training settings
epochs: 100
batch: 16
imgsz: 640
device: cuda

# Optimizer
optimizer: AdamW
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0

# Early stopping
patience: 50
```

### Training Tips

**Batch Size:**
- GPU memory permitting, larger batches (32-64) train faster
- Smaller batches (4-8) may generalize better for small datasets
- Use batch=16 as default

**Image Size:**
- imgsz=640 is standard for YOLOv8
- Larger (imgsz=1280) for small objects, but slower
- Smaller (imgsz=416) for faster inference, but lower accuracy

**Epochs:**
- 100 epochs minimum for small datasets
- 300+ epochs for large datasets
- Use early stopping (patience=50) to prevent overfitting

**Model Selection:**
- **yolov8n**: Fastest inference, lowest accuracy (recommended for edge devices)
- **yolov8s**: Good balance of speed and accuracy
- **yolov8m**: Better accuracy, slower inference
- **yolov8l/x**: Best accuracy, slowest (server deployment only)

## Directory Structure

```
training/
├── cli.py                    # Training CLI
├── configs/
│   └── hyperparameters.yaml  # Training hyperparameters
├── runs/                     # Training runs (gitignored)
│   ├── v1_baseline/
│   │   ├── weights/
│   │   │   ├── best.pt      # Best model
│   │   │   └── last.pt      # Latest checkpoint
│   │   ├── results.csv       # Training metrics
│   │   ├── confusion_matrix.png
│   │   ├── results.png
│   │   ├── training_config.json
│   │   └── metrics/
│   │       ├── metrics_train.json
│   │       ├── metrics_val.json
│   │       └── metrics_test.json
│   └── v2_improved/
│       └── ...
└── README.md
```

## Logging and Monitoring

Training metrics are logged to:
- **CSV**: `training/runs/{version}/results.csv`
- **TensorBoard**: `training/runs/{version}/` (if enabled)
- **Plots**: Confusion matrix, PR curves, loss curves

View TensorBoard:
```bash
tensorboard --logdir training/runs
```

## Experiment Tracking

Keep track of experiments in a training log:

```yaml
# training/experiment_log.yaml
experiments:
  - version: v1_baseline
    date: 2025-01-15
    model: yolov8n
    dataset: tiles_yolo_v1
    epochs: 100
    batch: 16
    map50: 0.842
    notes: |
      Baseline model with default hyperparameters.
      Good performance on pine/worker classes.
      Object class needs improvement.

  - version: v2_augmented
    date: 2025-01-20
    model: yolov8n
    dataset: tiles_yolo_v1
    epochs: 150
    batch: 16
    map50: 0.879
    notes: |
      Increased augmentation (mosaic=1.0, mixup=0.15).
      Better generalization on object class.
      15 FPS on RTX 3080.
```

## Additional Scripts

The training module also includes:

- **`train_yolo_pina_detector.py`** - Standalone training script (legacy)

**Note:** For most use cases, prefer the `cli.py train` command over the standalone script. The standalone script is preserved for backwards compatibility.

## Best Practices

1. **Baseline First**: Train a baseline model with default settings
2. **Version Everything**: Use clear version names (v1_baseline, v2_augmented)
3. **Track Experiments**: Document what changed between versions
4. **Evaluate on Test**: Always report final metrics on held-out test set
5. **Save Configs**: Keep training_config.json with each run
6. **Monitor Overfitting**: Watch val loss vs train loss
