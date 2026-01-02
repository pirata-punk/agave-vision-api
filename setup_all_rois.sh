#!/bin/bash
#
# ROI Setup Script for All Cameras
#
# This script will guide you through selecting ROI points for each camera.
# For each camera, an interactive window will open where you can:
#   - Left-click to add polygon points
#   - Right-click to remove last point
#   - Press 'N' to finish current ROI and start a new one
#   - Press 'S' to save and continue to next camera
#   - Press 'Q' to skip current camera
#

set -e

PROJECT_ROOT="/Users/jvmx/Documents/dev/agave-vision-api"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Agave Vision - ROI Setup for All Cameras"
echo "=========================================="
echo ""
echo "This script will help you define forbidden zones (ROIs) for all 8 cameras."
echo "You'll be prompted to select ROI polygons for each camera one by one."
echo ""
echo "Instructions:"
echo "  • LEFT CLICK  - Add point to polygon"
echo "  • RIGHT CLICK - Remove last point"
echo "  • C           - Clear all points"
echo "  • N           - Finish current ROI and start new one"
echo "  • S           - Save ROIs and continue to next camera"
echo "  • ENTER       - Save and continue to next camera"
echo "  • Q/ESC       - Skip this camera"
echo ""
read -p "Press ENTER to start..."
echo ""

# Camera configurations: "camera_id|image_path|camera_name"
declare -a CAMERAS=(
    "cam_nave3_hornos_a_cam3|data/videos/Video Volcador A Nave 3 CAM 3  PI„AS/NAVE 3_HORNOS A CAM 3_20250923162408_20250923165118.jpg|Nave 3 Hornos A CAM 3"
    "cam_nave3_hornos_a|data/videos/Video Volcador A Nave 3 HORNOS  PI„AS/NAVE 3_ HORNOS A_20250923162907_20250923165519.jpg|Nave 3 Hornos A"
    "cam_nave3_hornos_b|data/videos/Video Volcador B Nave 3 HORNOS  PI„AS/NAVE 3_ HORNOS B_20250923125053_20250923131505.jpg|Nave 3 Hornos B"
    "cam_nave3_hornos_b_cam3|data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.jpg|Nave 3 Hornos B CAM 3"
    "cam_nave4_difusor_a|data/videos/Video Volcador A Nave 4 Personas/NAVE 4_DIFUSOR A_20250929112706_20250929113002.jpg|Nave 4 Difusor A"
    "cam_nave4_difusor_a_cam3|data/videos/Video Volcador A CAM 3 Nave 4 Personas/NAVE 4_DIFUSORES LADO A CAM 3_20250929120626_20250929122803.jpg|Nave 4 Difusores Lado A CAM 3"
    "cam_nave4_difusor_b|data/videos/Video Volcador B Nave 4 PI„AS/NAVE 4_DIFUSOR  B_20250923180631_20250923180839.jpg|Nave 4 Difusor B"
    "cam_nave4_difusor_b_cam3|data/videos/Video Volcador B Nave 4 CAM 3  PI„AS/NAVE 4_DIFUSORES LADO B CAM 3_20250923215123_20250923215648.jpg|Nave 4 Difusores Lado B CAM 3"
)

TOTAL_CAMERAS=${#CAMERAS[@]}
CURRENT=1

for camera_config in "${CAMERAS[@]}"; do
    IFS='|' read -r camera_id image_path camera_name <<< "$camera_config"

    echo "=========================================="
    echo "Camera $CURRENT/$TOTAL_CAMERAS: $camera_name"
    echo "=========================================="
    echo "Camera ID: $camera_id"
    echo "Image: $image_path"
    echo ""

    # Check if image exists
    if [ ! -f "$image_path" ]; then
        echo "⚠️  WARNING: Image not found: $image_path"
        echo "Skipping this camera..."
        echo ""
        ((CURRENT++))
        continue
    fi

    echo "📐 Opening ROI selector..."
    echo ""

    # Run ROI selector
    python examples/roi_selector.py \
        --video "$image_path" \
        --camera-id "$camera_id" \
        --config "configs/rois.yaml" \
        --frame 0

    echo ""
    echo "✓ ROI configuration saved for $camera_name"
    echo ""

    if [ $CURRENT -lt $TOTAL_CAMERAS ]; then
        read -p "Press ENTER to continue to next camera..."
        echo ""
    fi

    ((CURRENT++))
done

echo "=========================================="
echo "  ✓ ROI Setup Complete!"
echo "=========================================="
echo ""
echo "All camera ROIs have been configured and saved to: configs/rois.yaml"
echo ""
echo "Next steps:"
echo "  1. Review the configuration: cat configs/rois.yaml"
echo "  2. Test with live demo:"
echo "     ./run_demo.sh"
echo ""
echo "  Or run demo with specific camera:"
echo "     python examples/live_demo.py \\"
echo "         --video \"<video_path>\" \\"
echo "         --roi-config configs/rois.yaml \\"
echo "         --camera-id <camera_id>"
echo ""
