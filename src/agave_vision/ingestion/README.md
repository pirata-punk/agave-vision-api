# Data Ingestion Module

Handles both **static data processing** (videos → frames → tiles → dataset) and **live streaming** (RTSP ingestion).

## Overview

The ingestion module provides a complete pipeline for preparing training data from raw videos to YOLO-ready datasets.

## Static Data Pipeline

```
Videos → Frames → Tiles → Labels → YOLO Dataset
```

### 1. Extract Frames from Videos

```bash
python ingestion/cli.py extract-frames \
    --video-dir data/videos \
    --output-dir data/frames \
    --sample-rate 30 \
    --dedup-threshold 0.95 \
    --min-sharpness 100.0
```

**What it does:**
- Extracts every Nth frame from videos
- Deduplicates similar frames using histogram comparison
- Filters blurry frames based on sharpness threshold
- Saves frames as JPEG files

**Output:**
- `data/frames/*.jpg` - Extracted frames
- `data/frames/frames_manifest.json` - Frame metadata

### 2. Generate Tiles

```bash
python ingestion/cli.py generate-tiles \
    --frames-dir data/frames \
    --output-dir data/tiles_pool \
    --tile-size 640 \
    --overlap 128
```

**What it does:**
- Generates fixed-size tiles using sliding window
- Creates 640x640 patches with configurable overlap
- Filters edge tiles with too much padding

**Output:**
- `data/tiles_pool/*.jpg` - Generated tiles
- `data/tiles_pool/metadata.json` - Tile metadata

### 3. Manual Labeling (External Tool)

Use a labeling tool (e.g., Roboflow, LabelImg, CVAT) to annotate tiles:

1. Label tiles in `data/tiles_pool/`
2. Export labels in YOLO format (.txt files)
3. Organize into rounds: `data/tiles_pool/tiles_man/tiles_round{1..4}/`
4. Each round should have `images/` and `labels/` subdirectories

### 4. Build YOLO Dataset

```bash
python ingestion/cli.py build-dataset \
    --rounds-dir data/tiles_pool/tiles_man \
    --output-dir data/tiles_yolo \
    --classes object pine worker \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 123
```

**What it does:**
- Collects labeled images from all rounds
- Splits into train/val/test sets (reproducible with seed)
- Copies files to YOLO dataset structure
- Generates `configs/yolo_data.yaml`

**Output:**
- `data/tiles_yolo/images/{train,val,test}/` - Images
- `data/tiles_yolo/labels/{train,val,test}/` - Labels
- `configs/yolo_data.yaml` - YOLO dataset configuration
- `data/tiles_yolo/metadata.json` - Dataset metadata

### Complete Pipeline

Run all steps at once (except labeling):

```bash
python ingestion/cli.py pipeline
```

## Live Streaming

Live RTSP streaming is handled by the production Stream Manager service.

See: `src/agave_vision/services/stream_manager/`

## Configuration

### Video Processing Config

Create `ingestion/configs/video_processing.yaml`:

```yaml
sample_rate: 30  # Extract every 30th frame
dedup_threshold: 0.95  # Similarity threshold
min_sharpness: 100.0  # Minimum sharpness score
```

### Dataset Builder Config

Create `ingestion/configs/dataset_builder.yaml`:

```yaml
class_names:
  - object
  - pine
  - worker

split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15

random_seed: 123
allow_empty_labels: true
```

## Programmatic Usage

```python
from agave_vision.ingestion.static.video_processor import VideoProcessor
from agave_vision.ingestion.static.tile_generator import TileGenerator
from agave_vision.ingestion.static.dataset_builder import DatasetBuilder

# Extract frames
processor = VideoProcessor(
    video_dir="data/videos",
    output_dir="data/frames"
)
frames_metadata = processor.process_all_videos()

# Generate tiles
generator = TileGenerator(
    frames_dir="data/frames",
    output_dir="data/tiles_pool"
)
tiles_metadata = generator.generate_all_tiles()

# Build dataset
builder = DatasetBuilder(
    rounds_dir="data/tiles_pool/tiles_man",
    output_dir="data/tiles_yolo",
    class_names=["object", "pine", "worker"]
)
dataset_metadata = builder.build_dataset()
```

## Directory Structure

```
data/
├── videos/              # Raw video files (.mp4, .avi)
├── frames/              # Extracted frames
│   └── frames_manifest.json
├── tiles_pool/          # Generated tiles
│   ├── metadata.json
│   └── tiles_man/       # Manually labeled tiles
│       ├── tiles_round1/
│       │   ├── images/
│       │   └── labels/
│       ├── tiles_round2/
│       ├── tiles_round3/
│       └── tiles_round4/
└── tiles_yolo/          # Final YOLO dataset
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── metadata.json
```

## Additional Scripts

The ingestion module also includes standalone scripts for specific tasks:

- **`extract_frames.py`** - Standalone frame extraction (legacy)
- **`generate_tiles.py`** - Standalone tile generation (legacy)
- **`clean_tiles.py`** - Clean and filter generated tiles
- **`split_rounds_from_clean.py`** - Organize tiles into rounds
- **`standardize_round_filenames.py`** - Standardize tile filenames
- **`scan_frames_metadata.py`** - Scan and analyze frame metadata
- **`build_yolo_dataset.py`** - Standalone dataset builder (legacy)

**Note:** For most use cases, prefer the `cli.py` commands over standalone scripts. The standalone scripts are preserved for backwards compatibility and specific edge cases.

## Tips

- **Frame sampling**: Use higher sample rates (60-120) for fast-moving videos
- **Tile overlap**: More overlap (256) captures objects at tile edges better
- **Sharpness threshold**: Increase (150-200) if videos are high quality
- **Class ordering**: MUST match YOLO label IDs (0: object, 1: pine, 2: worker)
