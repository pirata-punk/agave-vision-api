#!/bin/bash
#
# Quick Single Camera Test
#
# Tests a single camera with ROI alerts to validate Phase 1
#

PROJECT_ROOT="/Users/jvmx/Documents/dev/agave-vision-api"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Phase 1 - Single Camera Test"
echo "=========================================="
echo ""
echo "Testing: Nave 3 Hornos B CAM 3 (has 2 ROI zones)"
echo ""
echo "What to look for:"
echo "  ✓ Yellow polygons (2 ROI zones)"
echo "  ✓ Green boxes for pines (allowed)"
echo "  ✓ Orange boxes for workers (allowed)"
echo "  ✓ Red thick boxes for objects (ALERT)"
echo "  ✓ '⚠ ALERT!' labels on violations"
echo "  ✓ Alert count in top-right corner"
echo ""
echo "Controls:"
echo "  SPACE - Pause/Resume"
echo "  Q/ESC - Quit"
echo "  S     - Screenshot"
echo "  R     - Reset stats"
echo ""
read -p "Press ENTER to start demo..."

python examples/live_demo.py \
    --video "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3

echo ""
echo "=========================================="
echo "Test complete!"
echo ""
echo "Did you see the expected behavior?"
echo "  - Yellow ROI polygons: YES / NO"
echo "  - Red boxes for alerts: YES / NO"
echo "  - Alert count increasing: YES / NO"
echo ""
echo "If everything looked correct, Phase 1 is validated!"
echo "Ready to proceed to Phase 2."
echo "=========================================="
