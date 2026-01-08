#!/bin/bash
# Launcher script for Agave Vision Interactive Demo

cd "$(dirname "$0")/.." || exit 1

echo "🚀 Starting Agave Vision Interactive Demo..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if video file exists
VIDEO_FILE="demo/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

# Check if model exists
MODEL_FILE="models/agave-industrial-vision-v1.0.0.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: Model file not found: $MODEL_FILE"
    exit 1
fi

# Run demo
python demo/interactive_demo.py

echo ""
echo "✨ Demo completed"
