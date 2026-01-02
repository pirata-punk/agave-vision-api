# ROI Setup Guide - All Cameras

Complete guide for setting up Region of Interest (ROI) zones for all 8 cameras in the Agave Vision system.

## 📋 Detected Cameras

| # | Camera Name | Camera ID | Resolution |
|---|-------------|-----------|------------|
| 1 | Nave 3 Hornos A CAM 3 | `cam_nave3_hornos_a_cam3` | 2560x1440 |
| 2 | Nave 3 Hornos A | `cam_nave3_hornos_a` | 2688x1520 |
| 3 | Nave 3 Hornos B | `cam_nave3_hornos_b` | 2560x1440 |
| 4 | Nave 3 Hornos B CAM 3 | `cam_nave3_hornos_b_cam3` | 2560x1440 |
| 5 | Nave 4 Difusor A | `cam_nave4_difusor_a` | 2688x1520 |
| 6 | Nave 4 Difusores Lado A CAM 3 | `cam_nave4_difusor_a_cam3` | 2560x1440 |
| 7 | Nave 4 Difusor B | `cam_nave4_difusor_b` | 2560x1440 |
| 8 | Nave 4 Difusores Lado B CAM 3 | `cam_nave4_difusor_b_cam3` | 2560x1440 |

## 🚀 Quick Start

### Step 1: Preview Cameras
```bash
./preview_cameras.sh
```

This will show you all 8 cameras with their IDs and verify all images are available.

### Step 2: Run ROI Setup
```bash
./setup_all_rois.sh
```

This will guide you through selecting ROI points for each camera, one by one.

## 🎮 Interactive Controls

When the ROI selector window opens for each camera:

| Control | Action |
|---------|--------|
| **LEFT CLICK** | Add point to polygon |
| **RIGHT CLICK** | Remove last point |
| **C** | Clear all points for current ROI |
| **N** | Finish current ROI and start a new one |
| **S** | Save all ROIs and continue to next camera |
| **ENTER** | Save and continue to next camera |
| **Q** or **ESC** | Skip this camera (don't save) |

## 📐 ROI Selection Best Practices

### 1. Define Forbidden Zones
Mark areas where **only pines and workers** are allowed. Any other object (tires, logs, debris, etc.) in these zones will trigger an alert.

### 2. Click Polygon Corners
- Click at each corner of the forbidden zone
- The polygon will be drawn in real-time (yellow lines)
- Minimum 3 points required for a valid polygon

### 3. Multiple ROIs per Camera
- You can define multiple forbidden zones for the same camera
- After finishing one ROI, press **N** to start another
- Give each ROI a descriptive name when prompted (e.g., "loading_zone", "conveyor_area")

### 4. Naming Convention
When prompted for ROI name, use descriptive names:
- `loading_zone` - Where materials are loaded
- `conveyor_belt` - Conveyor belt area
- `danger_zone_1`, `danger_zone_2` - Multiple danger areas
- `personnel_area` - Worker-only zones

### 5. Visual Feedback
- 🔵 **Blue dots** - Individual polygon points
- 🟡 **Yellow lines** - Current polygon being drawn
- 🔴 **Red filled area** - Completed/saved ROIs

## 📂 Output

The script will create/update `configs/rois.yaml` with this structure:

```yaml
cameras:
  - camera_id: cam_nave3_hornos_a_cam3
    forbidden_rois:
      - name: loading_zone
        points:
          - [245, 180]
          - [580, 190]
          - [620, 450]
          - [210, 440]
      - name: danger_zone_1
        points:
          - [100, 500]
          - [700, 500]
          - [700, 600]
          - [100, 600]
    allowed_classes: [pine, worker]
    alert_classes: [object]
    strict_mode: true

  - camera_id: cam_nave3_hornos_a
    # ... (next camera configuration)
```

## ✅ After Setup

### Verify Configuration
```bash
cat configs/rois.yaml
```

### Test with Live Demo
Test a specific camera:
```bash
python examples/live_demo.py \
    --video "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3
```

You should see:
- ✅ Yellow polygons showing forbidden ROI zones
- ✅ Red bounding boxes for detections that would trigger alerts
- ✅ "⚠ ALERT!" labels on forbidden detections
- ✅ Alert count in the overlay

## 🎯 Phase 1 Alert Logic

With **strict_mode: true** (default), the system will:

✅ **Allow in forbidden zones:**
- Pines (`pine`)
- Workers (`worker`)

⚠️ **Alert on in forbidden zones:**
- Objects (`object`)
- Unknown/low-confidence detections
- Any other class not in `allowed_classes`

This **whitelist approach** eliminates the need to pre-train on specific forbidden objects (tires, logs, debris, etc.).

## 🔧 Editing ROIs Later

### Re-run for specific camera:
```bash
python examples/roi_selector.py \
    --video "data/videos/<camera_directory>/<image.jpg>" \
    --camera-id <camera_id> \
    --config configs/rois.yaml
```

### Manual editing:
Edit `configs/rois.yaml` directly to adjust polygon points.

## 📊 Configuration Options

Each camera in `configs/rois.yaml` supports:

| Field | Type | Description |
|-------|------|-------------|
| `camera_id` | string | Unique camera identifier |
| `forbidden_rois` | list | List of forbidden zone polygons |
| `forbidden_rois[].name` | string | ROI zone name |
| `forbidden_rois[].points` | list | Polygon points [[x,y], ...] |
| `allowed_classes` | list | Classes allowed in forbidden zones |
| `alert_classes` | list | (Legacy) Classes that trigger alerts |
| `strict_mode` | boolean | **NEW:** Alert on anything NOT in allowed_classes |

## 🚨 Troubleshooting

### Window doesn't appear
- Ensure you're running in a GUI environment (not SSH)
- Check that OpenCV is installed: `pip install opencv-python`

### Image not loading
- Verify image path exists
- Check file permissions
- Try opening image manually: `open "<image_path>"`

### ROI not saving
- Ensure you have at least 3 points before pressing 'S'
- Check write permissions for `configs/` directory
- Press 'S' or 'ENTER' to save (not just 'Q')

### Polygon points not precise
- Use the `--frame` parameter to select a clearer frame
- Zoom in on the image before clicking (if your viewer supports it)
- Click carefully at exact corner positions

## 📞 Need Help?

- Interactive tool controls: Press any key in the ROI selector window to see controls
- Re-run specific camera: See "Editing ROIs Later" section above
- Manual configuration: Edit `configs/rois.yaml` directly

---

**Next Step:** After completing ROI setup, Phase 1 will be complete! 🎉

Move on to testing the new alert logic with the live demo.
