# Model Configuration Guide

**Model:** `agave-industrial-vision-v1.0.0.pt`
**Last Updated:** 2026-01-06

---

## Overview

The Agave Vision API uses a centralized model configuration system that makes it easy to update the default model across the entire project. All model paths are managed through `configs/model.yaml`.

### Key Benefits

✅ **Single source of truth** - Update one file to change the model everywhere
✅ **Version management** - Track multiple model versions
✅ **Easy rollback** - Switch between model versions instantly
✅ **No code changes** - Update model without modifying Python code

---

## Quick Start

### Using Default Model (Recommended)

```python
from agave_vision.ml_api import AgaveVisionML

# Automatically uses model from configs/model.yaml
ml = AgaveVisionML()

# Or with ROI configuration
ml = AgaveVisionML(roi_config_path="configs/rois.yaml")
```

### Using Custom Model

```python
# Override with custom model path
ml = AgaveVisionML(model_path="models/custom-model.pt")
```

### Getting Model Info

```python
from agave_vision.config import get_default_model_path, get_model_info

# Get current default model path
model_path = get_default_model_path()
print(f"Default model: {model_path}")
# Output: models/agave-industrial-vision-v1.0.0.pt

# Get model metadata
info = get_model_info()
print(f"Model: {info['name']} v{info['version']}")
print(f"Architecture: {info['architecture']}")
print(f"Classes: {info['classes']}")
```

---

## Configuration File

**Location:** `configs/model.yaml`

```yaml
# Default model path (relative to project root)
default_model: "models/agave-industrial-vision-v1.0.0.pt"

# Model metadata
model_info:
  name: "Agave Industrial Vision"
  version: "1.0.0"
  architecture: "YOLOv8n"
  classes:
    - object
    - pine
    - worker
  num_classes: 3
  description: "YOLOv8 nano model trained for industrial surveillance"
  trained_date: "2025-12"

# Default inference parameters
inference:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  image_size: 640
  device: "cpu"

# Model versioning
versions:
  v1.0.0:
    path: "models/agave-industrial-vision-v1.0.0.pt"
    description: "Initial production model"
    performance:
      map50: 0.842
      map50_95: 0.678
```

---

## Updating the Model

### Scenario 1: Deploy New Model Version

When you train a new model and want to deploy it:

1. **Add new model file:**
   ```bash
   cp path/to/new-model.pt models/agave-industrial-vision-v2.0.0.pt
   ```

2. **Update `configs/model.yaml`:**
   ```yaml
   # Change this line
   default_model: "models/agave-industrial-vision-v2.0.0.pt"

   # Add version entry
   versions:
     v2.0.0:
       path: "models/agave-industrial-vision-v2.0.0.pt"
       description: "Improved detection accuracy"
       performance:
         map50: 0.891
         map50_95: 0.734
   ```

3. **Test the change:**
   ```bash
   python -c "from agave_vision.ml_api import AgaveVisionML; ml = AgaveVisionML(); print(ml.model_path)"
   # Should print: models/agave-industrial-vision-v2.0.0.pt
   ```

4. **Done!** All code automatically uses the new model.

### Scenario 2: Rollback to Previous Version

To rollback to an earlier model:

1. **Update `configs/model.yaml`:**
   ```yaml
   # Change back to previous version
   default_model: "models/agave-industrial-vision-v1.0.0.pt"
   ```

2. **Verify:**
   ```bash
   python -c "from agave_vision.config import get_default_model_path; print(get_default_model_path())"
   ```

### Scenario 3: A/B Testing Multiple Models

Use the versioning system for A/B testing:

```python
from agave_vision.config import get_model_path_for_version
from agave_vision.ml_api import AgaveVisionML

# Test version A
model_a_path = get_model_path_for_version("v1.0.0")
ml_a = AgaveVisionML(model_path=model_a_path)

# Test version B
model_b_path = get_model_path_for_version("v2.0.0")
ml_b = AgaveVisionML(model_path=model_b_path)

# Compare results...
```

---

## Model Naming Convention

Follow this naming pattern for consistency:

```
agave-<domain>-<descriptor>-v<major>.<minor>.<patch>.pt
```

**Examples:**
- `agave-industrial-vision-v1.0.0.pt` - Current production model
- `agave-industrial-vision-v1.1.0.pt` - Minor improvement
- `agave-forestry-detection-v1.0.0.pt` - Domain-specific variant
- `agave-industrial-vision-v2.0.0.pt` - Major architecture change

**Version Semantics:**
- **Major (v2.0.0):** Breaking changes, new architecture, significant accuracy shifts
- **Minor (v1.1.0):** Improvements, retraining on more data, same architecture
- **Patch (v1.0.1):** Bug fixes, model quantization, minor tweaks

---

## API Reference

### `get_default_model_path()`
Returns the default model path from configuration.

```python
from agave_vision.config import get_default_model_path

model_path = get_default_model_path()
# Returns: "models/agave-industrial-vision-v1.0.0.pt"
```

### `get_model_info()`
Returns model metadata dictionary.

```python
from agave_vision.config import get_model_info

info = get_model_info()
# Returns: {
#   "name": "Agave Industrial Vision",
#   "version": "1.0.0",
#   "architecture": "YOLOv8n",
#   "classes": ["object", "pine", "worker"],
#   ...
# }
```

### `get_inference_defaults()`
Returns default inference parameters.

```python
from agave_vision.config import get_inference_defaults

defaults = get_inference_defaults()
# Returns: {
#   "confidence_threshold": 0.25,
#   "iou_threshold": 0.45,
#   "image_size": 640,
#   "device": "cpu"
# }
```

### `get_model_path_for_version(version: str)`
Get model path for specific version.

```python
from agave_vision.config import get_model_path_for_version

path = get_model_path_for_version("v1.0.0")
# Returns: "models/agave-industrial-vision-v1.0.0.pt"
```

---

## Integration Examples

### Example 1: Flask API

```python
from flask import Flask
from agave_vision.ml_api import AgaveVisionML

app = Flask(__name__)

# Initialize once on startup (uses default model)
ml = AgaveVisionML(roi_config_path="configs/rois.yaml")

@app.route("/detect", methods=["POST"])
def detect():
    image = request.files["image"]
    result = ml.predict_frame(image.read())
    return jsonify(result)
```

### Example 2: FastAPI Service

```python
from fastapi import FastAPI
from agave_vision.ml_api import AgaveVisionML
from agave_vision.config import get_model_info

app = FastAPI()
ml = AgaveVisionML()

@app.get("/model-info")
def model_info():
    return get_model_info()

@app.post("/infer")
def infer(image: UploadFile):
    result = ml.predict_frame(image.file.read())
    return result
```

### Example 3: Batch Processing

```python
from agave_vision.ml_api import AgaveVisionML
import cv2
from pathlib import Path

# Initialize with default model
ml = AgaveVisionML()

# Process all images in directory
for img_path in Path("images/").glob("*.jpg"):
    image = cv2.imread(str(img_path))
    result = ml.predict_frame(image)
    print(f"{img_path.name}: {len(result['detections'])} detections")
```

---

## Services Configuration

The production services (`services/`) also use model configuration.

**File:** `configs/services.yaml`

```yaml
inference:
  model_path: "models/agave-industrial-vision-v1.0.0.pt"
  confidence: 0.25
  device: "cuda"
```

**To update model for services:**
1. Update `configs/services.yaml` with new model path
2. Restart services:
   ```bash
   docker compose restart inference-api stream-manager
   ```

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model file not found: models/agave-industrial-vision-v1.0.0.pt
```

**Solution:**
- Verify model file exists: `ls -lh models/`
- Check path in `configs/model.yaml`
- Ensure path is relative to project root

### Wrong Model Loading

```python
# Check what model is being loaded
from agave_vision.config import get_default_model_path
print(get_default_model_path())
```

### Config Not Found

If `configs/model.yaml` is missing, the system falls back to:
```python
default_model = "models/agave-industrial-vision-v1.0.0.pt"
```

---

## Best Practices

1. ✅ **Always update `configs/model.yaml`** - Don't hardcode model paths in code
2. ✅ **Version your models** - Use semantic versioning in filenames
3. ✅ **Track performance** - Document metrics in `configs/model.yaml`
4. ✅ **Test before production** - Verify new model with test set before deploying
5. ✅ **Keep old versions** - Don't delete previous model files (for rollback)
6. ✅ **Document changes** - Add notes in `versions` section of config

---

## Migration from `best.pt`

The old `best.pt` naming has been replaced with versioned names:

**Old way:**
```python
ml = AgaveVisionML(model_path="models/best.pt")  # ❌ Unclear
```

**New way:**
```python
ml = AgaveVisionML()  # ✅ Uses configured model
# or
ml = AgaveVisionML(model_path="models/agave-industrial-vision-v1.0.0.pt")
```

**Benefits:**
- Clear versioning
- Easy to track which model is deployed
- Supports multiple models simultaneously
- Professional naming convention

---

## Summary

The centralized model configuration provides:

- **Single source of truth**: `configs/model.yaml`
- **Zero-code updates**: Change model without modifying Python files
- **Version management**: Track and switch between model versions
- **Easy rollback**: Revert to previous version instantly
- **Professional naming**: Clear, descriptive model filenames

**To update the model, just edit one file:** `configs/model.yaml`

Everything else updates automatically! 🎯
