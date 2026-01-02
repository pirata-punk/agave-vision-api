#!/bin/bash
#
# Test Alert System with Synthetic Objects
#
# Injects synthetic objects into ROI zones to trigger alerts
# Uses CV techniques to create realistic test scenarios
#

PROJECT_ROOT="/Users/jvmx/Documents/dev/agave-vision-api"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Alert Testing - Synthetic Objects"
echo "=========================================="
echo ""
echo "This test injects synthetic objects into ROI zones"
echo "to validate the alert system behavior."
echo ""
echo "Synthetic objects created using CV techniques:"
echo "  - Box/Crate (brown rectangle)"
echo "  - Tire/Barrel (dark circle)"
echo "  - Debris (irregular blob)"
echo ""
echo "These objects will be placed in the ROI zones"
echo "and should trigger alerts since they are NOT"
echo "pine or worker (the allowed classes)."
echo ""
echo "Controls during test:"
echo "  SPACE - Pause/Resume"
echo "  Q/ESC - Quit"
echo "  S     - Screenshot"
echo "  1     - Show only box objects"
echo "  2     - Show only tire objects"
echo "  3     - Show only debris objects"
echo ""
read -p "Press ENTER to start test..."
echo ""

# Default test - all object types in ROI center
python examples/test_alerts_with_synthetic_objects.py \
    --video "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3 \
    --object-type all \
    --position roi_center

echo ""
echo "=========================================="
echo "Test Complete!"
echo ""
echo "Expected behavior:"
echo "  ✓ Yellow ROI polygons visible"
echo "  ✓ Synthetic objects visible in ROI zones"
echo "  ✓ Red boxes around synthetic objects"
echo "  ✓ '⚠ ALERT!' labels on detections"
echo "  ✓ Alert count > 0"
echo ""
echo "If alerts were triggered, the system is working!"
echo "=========================================="
