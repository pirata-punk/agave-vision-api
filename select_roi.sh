#!/bin/bash
# Quick launch script for ROI selector

VIDEO_PATH="/Users/jvmx/Documents/dev/agave-vision-api/data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
CAMERA_ID="cam_nave3_hornos"

echo "📐 Launching ROI Selector..."
echo "Camera: $CAMERA_ID"
echo "Video: NAVE 3_HORNOS B CAM 3"
echo ""
echo "Instructions will appear in the window!"
echo ""

python examples/roi_selector.py \
    --video "$VIDEO_PATH" \
    --camera-id "$CAMERA_ID" \
    --frame 100

echo ""
echo "ROI selection finished!"
echo "Check configs/rois.yaml for the saved configuration"
