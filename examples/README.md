# Examples & Demo Scripts

Professional demonstration scripts for showcasing the Agave Vision object detection system.

## 🎯 Quick Setup - All Cameras (NEW!)

### Automated ROI Setup for All 8 Cameras

**Step 1: Preview all cameras**
```bash
./preview_cameras.sh
```

**Step 2: Set up ROI zones for all cameras**
```bash
./setup_all_rois.sh
```

This script will guide you through selecting forbidden zones for each of the 8 cameras:
1. Nave 3 Hornos A CAM 3
2. Nave 3 Hornos A
3. Nave 3 Hornos B
4. Nave 3 Hornos B CAM 3
5. Nave 4 Difusor A
6. Nave 4 Difusores Lado A CAM 3
7. Nave 4 Difusor B
8. Nave 4 Difusores Lado B CAM 3

The script opens an interactive window for each camera where you can define forbidden zones. All configurations are saved to `configs/rois.yaml`.

**See detailed guide:** [ROI Setup Guide](../docs/roi_setup_guide.md)

---

## 📐 ROI Selector (Interactive Configuration)

### Quick Start

**Configure ROI for your camera:**
```bash
# From project root
./select_roi.sh
```

**Or run directly:**
```bash
python examples/roi_selector.py --video "path/to/video.mp4" --camera-id cam_nave3_hornos
```

### How It Works

The ROI Selector opens an interactive GUI where you can:

1. **Click points** on a video frame to define polygon boundaries
2. **See the polygon** being drawn in real-time
3. **Create multiple ROIs** for the same camera
4. **Save directly** to `configs/rois.yaml`

### Step-by-Step Guide

1. **Run the selector** - A window opens showing a frame from your video
2. **Click to add points** - Left-click to add corners of your ROI polygon
3. **Right-click to undo** - Remove the last point if you make a mistake
4. **Press 'N'** - Finish current ROI and start drawing a new one
5. **Press 'S' or ENTER** - Save all ROIs to config file

### Interactive Controls

| Action | Control |
|--------|---------|
| Add point to polygon | **LEFT CLICK** |
| Remove last point | **RIGHT CLICK** |
| Clear all points | **C** |
| Finish ROI & start new | **N** |
| Save to config | **S** |
| Save and quit | **ENTER** |
| Quit without saving | **Q** or **ESC** |

### Visual Feedback

- 🔵 **Blue dots** - Individual points you've clicked
- 🟡 **Yellow lines** - Current polygon being drawn
- 🔴 **Red filled area** - Completed/saved ROIs

### Usage Examples

**Select ROI from specific frame:**
```bash
python examples/roi_selector.py \
    --video data/videos/video.mp4 \
    --camera-id cam_nave3_hornos \
    --frame 500
```

**Use custom config file:**
```bash
python examples/roi_selector.py \
    --video data/videos/video.mp4 \
    --camera-id cam1 \
    --config configs/custom_rois.yaml
```

**Different camera:**
```bash
python examples/roi_selector.py \
    --video data/videos/camera2.mp4 \
    --camera-id cam_nave4_difusor
```

### Output

The tool saves ROI configuration to `configs/rois.yaml`:

```yaml
cameras:
  - camera_id: cam_nave3_hornos
    forbidden_rois:
      - name: loading_zone
        points:
          - [245, 180]
          - [580, 190]
          - [620, 450]
          - [210, 440]
      - name: conveyor_belt
        points:
          - [100, 500]
          - [700, 500]
          - [700, 600]
          - [100, 600]
    allowed_classes: [pine, worker]
    alert_classes: [object]
```

### Tips

1. **Choose a clear frame** - Use `--frame` to select a frame where the area is clearly visible
2. **Define forbidden zones** - Mark areas where objects should NOT appear
3. **Multiple ROIs** - You can define several zones per camera
4. **Precision** - Click carefully at polygon corners for accurate boundaries
5. **Test it** - After saving, run the live demo to see ROIs in action

---

## 🎥 Live Demo

### Quick Start

**Run demo on specific video:**
```bash
# From project root
./run_demo.sh
```

**Or run directly:**
```bash
python examples/live_demo.py --video "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
```

### Features

- ✅ Real-time object detection visualization
- ✅ Bounding boxes with class labels and confidence scores
- ✅ Live FPS counter and statistics
- ✅ Detection counts per class (object, pine, worker)
- ✅ Professional overlay with statistics panel
- ✅ Keyboard controls for interaction
- ✅ Screenshot capture
- ✅ Video looping for continuous demo

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume playback |
| `Q` or `ESC` | Quit demo |
| `S` | Save screenshot |
| `R` | Reset statistics |

### Usage Examples

**Basic usage:**
```bash
python examples/live_demo.py --video path/to/video.mp4
```

**Custom model:**
```bash
python examples/live_demo.py \
    --video path/to/video.mp4 \
    --model models/custom/best.pt
```

**Adjust confidence threshold:**
```bash
python examples/live_demo.py \
    --video path/to/video.mp4 \
    --confidence 0.5
```

**Use webcam for live detection:**
```bash
python examples/live_demo.py --webcam
```

**Full HD display:**
```bash
python examples/live_demo.py \
    --video path/to/video.mp4 \
    --width 1920 \
    --height 1080
```

### Command-Line Options

```
Options:
  --video PATH           Path to video file
  --webcam               Use webcam instead of video file
  --model PATH           Path to trained model
                         (default: models/yolov8n_pina/exp/weights/best.pt)
  --confidence FLOAT     Confidence threshold (default: 0.25)
  --width INT            Display width (default: 1280)
  --height INT           Display height (default: 720)
  -h, --help             Show help message
```

## 📸 Demo Features

### Visual Elements

**Bounding Boxes:**
- 🔴 **Red** - Objects
- 🟢 **Green** - Pines
- 🟠 **Orange** - Workers

**Statistics Overlay:**
- Current FPS
- Frame counter
- Live detection count
- Total detections per class

### Screenshot Capture

Press `S` during demo to capture screenshots:
- Saves as `demo_screenshot_001.jpg`, `demo_screenshot_002.jpg`, etc.
- Includes all overlays and statistics
- Perfect for presentations

## 🎬 Other Demo Scripts

### demo_video_infer.py
Basic video inference with annotation saving.

```bash
python examples/demo_video_infer.py
```

### infer_alert.py
Lightweight inference with ROI-based alerting.

```bash
python examples/infer_alert.py
```

### realtime_yolo_stream.py
Real-time RTSP stream processing.

```bash
python examples/realtime_yolo_stream.py
```

## 💡 Tips for Best Demo Experience

1. **Lighting**: Ensure good video quality for best detection results
2. **Confidence**: Start with 0.25, increase to 0.5 for fewer false positives
3. **Display Size**: Use Full HD (1920x1080) for presentations
4. **Performance**: Close other applications for smooth FPS
5. **Screenshots**: Capture interesting detections for documentation

## 🚀 Presentation Mode

For live presentations:

```bash
# Use Full HD and higher confidence
python examples/live_demo.py \
    --video "path/to/demo/video.mp4" \
    --width 1920 \
    --height 1080 \
    --confidence 0.35
```

The video will automatically loop, making it perfect for continuous demonstrations!

## 📊 Performance

**Expected Performance:**
- **GPU (CUDA)**: 30-60 FPS on 1280x720 video
- **GPU (CUDA)**: 15-30 FPS on 1920x1080 video
- **CPU**: 5-15 FPS (not recommended for live demos)

**Optimization:**
- Ensure CUDA is available: Check with `nvidia-smi`
- Close other GPU-intensive applications
- Use lower resolution for smoother playback
