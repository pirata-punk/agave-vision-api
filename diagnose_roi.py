#!/usr/bin/env python3
"""
ROI Configuration Diagnostic Tool

Helps troubleshoot ROI loading issues
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("ROI Configuration Diagnostic")
print("=" * 60)
print()

# Test 1: Check if ROI config file exists
print("Test 1: ROI Config File")
print("-" * 60)
roi_config_path = Path("configs/rois.yaml")
if roi_config_path.exists():
    print(f"✓ Found: {roi_config_path}")
    print(f"  Size: {roi_config_path.stat().st_size} bytes")
else:
    print(f"✗ NOT FOUND: {roi_config_path}")
    sys.exit(1)
print()

# Test 2: Try loading with YAML
print("Test 2: YAML Parsing")
print("-" * 60)
try:
    import yaml
    with open(roi_config_path) as f:
        config = yaml.safe_load(f)
    print(f"✓ YAML loaded successfully")
    print(f"  Cameras found: {len(config.get('cameras', []))}")

    # List all cameras
    for cam in config.get('cameras', []):
        camera_id = cam.get('camera_id', 'UNKNOWN')
        num_rois = len(cam.get('forbidden_rois', []))
        print(f"    - {camera_id} ({num_rois} ROI zones)")
except Exception as e:
    print(f"✗ YAML parsing failed: {e}")
    sys.exit(1)
print()

# Test 3: Try loading with ROIManager
print("Test 3: ROIManager Loading")
print("-" * 60)
try:
    from agave_vision.core.roi import ROIManager

    roi_manager = ROIManager(roi_config_path)
    print(f"✓ ROIManager loaded successfully")
    print(f"  Cameras: {list(roi_manager.camera_rois.keys())}")
except Exception as e:
    print(f"✗ ROIManager failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 4: Check specific camera
print("Test 4: Target Camera (cam_nave3_hornos_b_cam3)")
print("-" * 60)
target_camera = "cam_nave3_hornos_b_cam3"
camera_roi = roi_manager.get_camera_rois(target_camera)

if camera_roi:
    print(f"✓ Camera '{target_camera}' found")
    print(f"  Forbidden zones: {len(camera_roi.forbidden_zones)}")
    print(f"  Allowed classes: {list(camera_roi.allowed_classes)}")
    print(f"  Alert classes: {list(camera_roi.alert_classes)}")
    print(f"  Strict mode: {camera_roi.strict_mode}")
    print()
    print("  ROI Zone Details:")
    for i, zone in enumerate(camera_roi.forbidden_zones, 1):
        print(f"    Zone {i}: {zone.name}")
        print(f"      Points: {len(zone.points)} vertices")
        print(f"      Coordinates: {zone.points.tolist()}")
else:
    print(f"✗ Camera '{target_camera}' NOT FOUND")
    print(f"  Available cameras: {list(roi_manager.camera_rois.keys())}")
    sys.exit(1)
print()

# Test 5: Check video file
print("Test 5: Video File")
print("-" * 60)
video_path = Path("data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4")
if video_path.exists():
    print(f"✓ Video found: {video_path}")
    print(f"  Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"✗ Video NOT FOUND: {video_path}")
print()

# Test 6: Check imports
print("Test 6: Required Imports")
print("-" * 60)
try:
    from agave_vision.core.inference import Detection
    print("✓ Detection class imported")

    import cv2
    print(f"✓ OpenCV imported (version: {cv2.__version__})")

    from ultralytics import YOLO
    print("✓ Ultralytics YOLO imported")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
print()

print("=" * 60)
print("✅ ALL DIAGNOSTICS PASSED")
print("=" * 60)
print()
print("ROI configuration is valid and should work.")
print()
print("If the demo still doesn't show ROI zones, please check:")
print("  1. Did you see console output about ROI loading?")
print("  2. Is the camera window actually opening?")
print("  3. Try running with verbose output:")
print()
print("     python examples/live_demo.py \\")
print("         --video \"data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4\" \\")
print("         --roi-config configs/rois.yaml \\")
print("         --camera-id cam_nave3_hornos_b_cam3")
print()
