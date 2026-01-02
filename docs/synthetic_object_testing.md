# Synthetic Object Testing for Alert Validation

## Overview

Since your production videos don't contain "object" class detections or unknown objects, this testing utility injects synthetic objects into video frames to validate the ROI alert system.

## How It Works

### CV Techniques Used

The test utility uses several computer vision techniques to create realistic test scenarios:

1. **Geometric Primitives**
   - `cv2.rectangle()` - Creates box/crate shapes
   - `cv2.circle()` - Creates tire/barrel shapes
   - `cv2.fillPoly()` - Creates irregular debris shapes

2. **Alpha Blending**
   - `cv2.addWeighted()` - Blends synthetic objects into real frames with transparency
   - Makes objects look naturally placed in the scene

3. **Template Scaling**
   - `cv2.resize()` - Allows size variation of synthetic objects

4. **ROI Geometry**
   - Point-in-polygon testing ensures objects are placed inside ROI zones
   - Calculates ROI centroid for placement

### Synthetic Object Types

Three types of synthetic objects are created:

| Type | Description | Color | Shape |
|------|-------------|-------|-------|
| Box | Simulates crates/boxes | Brown | Rectangle |
| Tire | Simulates tires/barrels | Dark gray | Circle with rings |
| Debris | Simulates debris/unknown | Grayish-brown | Irregular polygon |

All objects are:
- Semi-transparent (70% opacity)
- Classified as "object" class
- Placed inside ROI zones
- Should trigger alerts (not in allowed_classes)

## Usage

### Quick Test (Recommended)

```bash
./test_synthetic_alerts.sh
```

This runs a standard test with all 3 object types in ROI center position.

### Custom Test

```bash
python examples/test_alerts_with_synthetic_objects.py \
    --video "path/to/video.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id <camera_id> \
    --object-type <type> \
    --position <position>
```

**Parameters:**

- `--video`: Path to input video file
- `--roi-config`: Path to ROI configuration YAML
- `--camera-id`: Camera ID for ROI lookup
- `--object-type`: Type of object to inject
  - `box` - Only box/crate objects
  - `tire` - Only tire/barrel objects
  - `debris` - Only debris objects
  - `all` - All three types (default)
- `--position`: Placement strategy
  - `roi_center` - Center of ROI zones (default)
  - `random` - Random positions within ROI
  - `moving` - Objects move in circular patterns
- `--model`: (Optional) Path to YOLO model for real detections too

### Interactive Controls

While the test is running:

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume video |
| Q or ESC | Quit test |
| S | Save screenshot |
| 1 | Show only box objects |
| 2 | Show only tire objects |
| 3 | Show only debris objects |

## Expected Results

When running the synthetic object test, you should see:

### Visual Indicators

✅ **Yellow polygons** - ROI forbidden zones outlined
✅ **Synthetic objects** - Semi-transparent shapes overlaid on video
✅ **Red thick boxes** - Bounding boxes around synthetic objects
✅ **"⚠ ALERT!" labels** - Alert prefix on detection labels
✅ **Alert count** - Increasing counter in top-right overlay

### On-Screen Stats

The overlay displays:
- Frame count (current/total)
- Alert count (should be > 0)
- Number of synthetic objects injected
- Current object type

### Console Output

At test completion:
```
==========================================
FINAL RESULTS
==========================================
Total Alerts: 150
Frames Processed: 1000
Alert Rate: 15.00%
==========================================
```

## Validation Checklist

Use this checklist to validate the alert system:

- [ ] Synthetic objects are visible in the video
- [ ] Objects are placed inside yellow ROI zones
- [ ] Red bounding boxes appear around synthetic objects
- [ ] "⚠ ALERT!" prefix appears on labels
- [ ] Alert count increases frame by frame
- [ ] Alert count > 0 at end of test
- [ ] Different object types all trigger alerts
- [ ] Moving objects continue to trigger alerts

## Why This Test Is Important

Your production videos may not contain actual violations (objects in forbidden zones), which makes it difficult to validate that the alert system works correctly. This synthetic object injection approach:

1. **Guarantees test coverage** - Objects are always placed in ROI zones
2. **Validates whitelist logic** - Objects are NOT in allowed_classes, so they must alert
3. **Tests visual indicators** - Verifies red boxes, thick borders, and alert labels appear
4. **Confirms ROI geometry** - Tests that point-in-polygon detection works
5. **Provides quantitative metrics** - Alert count and rate can be measured

## Troubleshooting

### No alerts triggered

**Symptom:** Alert count stays at 0

**Solutions:**
1. Verify ROI zones are displayed (yellow polygons)
2. Check that synthetic objects appear in the video
3. Ensure objects are inside ROI zones (not outside)
4. Verify `strict_mode: true` in ROI config
5. Check `allowed_classes: [pine, worker]` excludes "object"

### Objects not visible

**Symptom:** No synthetic objects appear in video

**Solutions:**
1. Try increasing alpha value in code (make more opaque)
2. Verify ROI zones exist (need zones to place objects)
3. Check frame dimensions match ROI coordinates
4. Try different object types (use keys 1, 2, 3)

### All detections showing as alerts

**Symptom:** Even real pines/workers show alerts

**Solutions:**
1. This test doesn't use YOLO model by default
2. Add `--model` parameter to also show real detections
3. Real pines/workers should show green/orange, not red

## Advanced Usage

### Test with Real + Synthetic Detections

Combine YOLO detections with synthetic objects:

```bash
python examples/test_alerts_with_synthetic_objects.py \
    --video "path/to/video.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3 \
    --model models/yolov8n_pina/exp/weights/best.pt \
    --object-type all \
    --position moving
```

This will show:
- Green boxes for real pines (allowed, no alert)
- Orange boxes for real workers (allowed, no alert)
- Red boxes for synthetic objects (NOT allowed, alert!)

### Moving Object Test

Test continuous tracking:

```bash
python examples/test_alerts_with_synthetic_objects.py \
    --video "path/to/video.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3 \
    --object-type tire \
    --position moving
```

Objects will move in circular patterns through ROI zones, continuously triggering alerts.

## Technical Details

### Synthetic Detection Creation

Each synthetic object generates a `Detection` object:

```python
Detection(
    bbox=(x1, y1, x2, y2),      # Bounding box
    confidence=0.85,             # High confidence
    class_name="object",         # NOT in allowed_classes
    class_id=0,                  # Dummy ID
    center=(cx, cy),             # Center point
    is_unknown=False             # Known object type
)
```

### Alert Logic Flow

1. Inject synthetic object into frame (CV blending)
2. Create Detection with class_name="object"
3. Call `camera_roi.should_alert(detection)`
4. Returns `True` because:
   - Detection center is inside ROI zone (point-in-polygon)
   - class_name "object" NOT in allowed_classes `[pine, worker]`
   - strict_mode is enabled
5. Draw red box with "⚠ ALERT!" label
6. Increment alert counter

## Files Created

- `examples/test_alerts_with_synthetic_objects.py` - Main test utility
- `test_synthetic_alerts.sh` - Convenient test runner
- `docs/synthetic_object_testing.md` - This documentation

## Next Steps

After validating with synthetic objects:

1. ✅ Confirm alerts trigger correctly
2. ✅ Verify visual indicators are clear
3. ✅ Test all 8 cameras with synthetic objects
4. 📋 Document alert thresholds and rates
5. 📋 Proceed to Phase 2 (ML-Focused API Architecture)
