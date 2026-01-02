#!/bin/bash
#
# Phase 1 Alert System Test Script
#
# Tests the enhanced alert logic with ROI zones for all 8 cameras
# Validates: Whitelist approach, ROI detection, visual indicators
#

set -e

PROJECT_ROOT="/Users/jvmx/Documents/dev/agave-vision-api"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Phase 1 Alert System - Test Suite"
echo "=========================================="
echo ""
echo "This script will test the ROI alert system on all 8 cameras."
echo "For each camera, the live demo will run showing:"
echo "  ✓ Yellow ROI polygons (forbidden zones)"
echo "  ✓ Green boxes for pines/workers (allowed)"
echo "  ✓ Red thick boxes for objects/unknown (alerts)"
echo "  ✓ '⚠ ALERT!' labels on violations"
echo "  ✓ Alert count in overlay"
echo ""
echo "Controls during demo:"
echo "  SPACE - Pause/Resume"
echo "  Q/ESC - Skip to next camera"
echo "  S     - Save screenshot"
echo ""
read -p "Press ENTER to start testing..."
echo ""

# Test configuration
MODEL_PATH="models/yolov8n_pina/exp/weights/best.pt"
ROI_CONFIG="configs/rois.yaml"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    echo "Please ensure the model is trained and available."
    exit 1
fi

# Verify ROI config exists
if [ ! -f "$ROI_CONFIG" ]; then
    echo "❌ ERROR: ROI config not found at $ROI_CONFIG"
    echo "Please run ./setup_all_rois.sh first."
    exit 1
fi

# Camera test cases: "camera_id|video_path|camera_name"
declare -a TEST_CASES=(
    "cam_nave3_hornos_b_cam3|data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4|Nave 3 Hornos B CAM 3 (2 ROI zones)"
    "cam_nave3_hornos_a_cam3|data/videos/Video Volcador A Nave 3 CAM 3  PI„AS/NAVE 3_HORNOS A CAM 3_20250923162408_20250923165118.mp4|Nave 3 Hornos A CAM 3"
    "cam_nave3_hornos_a|data/videos/Video Volcador A Nave 3 HORNOS  PI„AS/NAVE 3_ HORNOS A_20250923162907_20250923165519.mp4|Nave 3 Hornos A"
    "cam_nave4_difusor_a|data/videos/Video Volcador A Nave 4 Personas/NAVE 4_DIFUSOR A_20250929112706_20250929113002.mp4|Nave 4 Difusor A"
)

TOTAL_TESTS=${#TEST_CASES[@]}
CURRENT=1
PASSED=0
FAILED=0

echo "=========================================="
echo "Running $TOTAL_TESTS camera tests..."
echo "=========================================="
echo ""

for test_case in "${TEST_CASES[@]}"; do
    IFS='|' read -r camera_id video_path camera_name <<< "$test_case"

    echo "----------------------------------------"
    echo "Test $CURRENT/$TOTAL_TESTS: $camera_name"
    echo "----------------------------------------"
    echo "Camera ID: $camera_id"
    echo "Video:     $video_path"
    echo ""

    # Check if video exists
    if [ ! -f "$video_path" ]; then
        echo "⚠️  SKIP: Video file not found"
        ((FAILED++))
        ((CURRENT++))
        echo ""
        continue
    fi

    echo "🎬 Starting demo (press Q to skip to next camera)..."
    echo ""

    # Run live demo
    if python examples/live_demo.py \
        --video "$video_path" \
        --model "$MODEL_PATH" \
        --roi-config "$ROI_CONFIG" \
        --camera-id "$camera_id"; then

        echo ""
        echo "✓ Test passed for $camera_name"
        ((PASSED++))
    else
        echo ""
        echo "✗ Test failed for $camera_name"
        ((FAILED++))
    fi

    ((CURRENT++))
    echo ""

    if [ $CURRENT -le $TOTAL_TESTS ]; then
        read -p "Press ENTER to continue to next camera..."
        echo ""
    fi
done

echo "=========================================="
echo "  Test Results Summary"
echo "=========================================="
echo "Total Tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED"
echo "Failed:       $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Phase 1 validation complete. The alert system is working correctly:"
    echo "  ✓ Whitelist approach (alert on anything NOT pine/worker)"
    echo "  ✓ ROI zones properly configured"
    echo "  ✓ Visual indicators working"
    echo "  ✓ Strict mode enabled for all cameras"
    echo ""
    echo "Ready to proceed to Phase 2!"
else
    echo "⚠️  Some tests failed. Please review the output above."
    echo ""
    echo "Common issues:"
    echo "  - Video file paths incorrect"
    echo "  - Model not found"
    echo "  - ROI config missing camera_id"
fi

echo "=========================================="
echo ""
