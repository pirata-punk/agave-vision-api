#!/bin/bash
# Quick launch script for live demo on archived video

VIDEO_PATH="/Users/jvmx/Documents/dev/agave-vision-api/data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
MODEL_PATH="models/yolov8n_pina/exp/weights/best.pt"

echo "🎥 Launching Agave Vision Live Demo..."
echo "Video: NAVE 3_HORNOS B CAM 3"
echo "Model: yolov8n_pina"
echo ""

python examples/live_demo.py \
    --video "$VIDEO_PATH" \
    --model "$MODEL_PATH" \
    --confidence 0.25 \
    --width 1280 \
    --height 720

echo ""
echo "Demo finished!"
