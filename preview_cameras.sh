#!/bin/bash
#
# Camera Preview Script
#
# Shows all detected cameras and their representative images
#

PROJECT_ROOT="/Users/jvmx/Documents/dev/agave-vision-api"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Detected Cameras - Preview"
echo "=========================================="
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
VALID_CAMERAS=0

echo "Found $TOTAL_CAMERAS cameras:"
echo ""

for i in "${!CAMERAS[@]}"; do
    camera_config="${CAMERAS[$i]}"
    IFS='|' read -r camera_id image_path camera_name <<< "$camera_config"

    camera_num=$((i + 1))

    printf "[$camera_num] %s\n" "$camera_name"
    printf "    Camera ID: %s\n" "$camera_id"
    printf "    Image:     %s\n" "$image_path"

    if [ -f "$image_path" ]; then
        # Get image dimensions
        image_info=$(python3 -c "import cv2; img=cv2.imread('$image_path'); print(f'{img.shape[1]}x{img.shape[0]}') if img is not None else print('Error')" 2>/dev/null || echo "Unknown")
        printf "    Status:    ✓ Found (%s)\n" "$image_info"
        ((VALID_CAMERAS++))
    else
        printf "    Status:    ✗ NOT FOUND\n"
    fi
    echo ""
done

echo "=========================================="
echo "Summary: $VALID_CAMERAS/$TOTAL_CAMERAS cameras have valid images"
echo "=========================================="
echo ""

if [ $VALID_CAMERAS -eq $TOTAL_CAMERAS ]; then
    echo "✓ All cameras ready for ROI setup!"
    echo ""
    echo "To start ROI selection, run:"
    echo "  ./setup_all_rois.sh"
elif [ $VALID_CAMERAS -gt 0 ]; then
    echo "⚠  Some cameras are missing images"
    echo "You can still proceed, but those cameras will be skipped."
    echo ""
    echo "To start ROI selection, run:"
    echo "  ./setup_all_rois.sh"
else
    echo "✗ No valid camera images found!"
    echo "Please check your data/videos directory."
fi
echo ""
