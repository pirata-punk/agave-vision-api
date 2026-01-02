# Phase 1 Validation Checklist

Complete validation guide for the enhanced alert system (Phase 1).

## 🎯 What We're Testing

The Phase 1 implementation introduced:
1. **Whitelist alert logic** - Alert on ANYTHING that is NOT pine/worker
2. **Unknown object detection** - Low-confidence detections flagged
3. **Enhanced alert events** - New fields: roi_name, violation_type, strict_mode
4. **ROI configuration** - 8 cameras with forbidden zones
5. **Visual demo** - Live demo with alert indicators

---

## 🧪 Quick Test (5 minutes)

### Option 1: Automated Test Suite

Run the automated test script that validates 4 cameras:

```bash
./test_phase1_alerts.sh
```

**What it tests:**
- ✅ ROI zones display correctly (yellow polygons)
- ✅ Alert detections show red boxes
- ✅ Normal detections show green/orange boxes
- ✅ Alert labels have "⚠ ALERT!" prefix
- ✅ Alert count displayed in overlay

### Option 2: Manual Quick Test

Test a single camera manually:

```bash
python examples/live_demo.py \
    --video "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4" \
    --roi-config configs/rois.yaml \
    --camera-id cam_nave3_hornos_b_cam3
```

---

## ✅ Validation Checklist

### Visual Indicators
When running the demo, verify you see:

- [ ] **Yellow polygons** outlining forbidden ROI zones
- [ ] **ROI labels** showing zone names (e.g., "ROI: cam_nave3_hornos_b_cam3_1")
- [ ] **Green boxes** for pine detections (allowed class)
- [ ] **Orange boxes** for worker detections (allowed class)
- [ ] **Red thick boxes** for object detections (alert-triggering)
- [ ] **"⚠ ALERT!"** prefix on labels for forbidden detections
- [ ] **Alert count** in top-right of overlay (e.g., "ALERTS: 15")

### Alert Logic Behavior

Test the whitelist approach:

- [ ] **Pine enters ROI** → Green box, NO alert, no red color
- [ ] **Worker enters ROI** → Orange box, NO alert, no red color
- [ ] **Object enters ROI** → Red thick box, ALERT label, alert count increases
- [ ] **Unknown object (low confidence) enters ROI** → Red box, ALERT label

### Configuration

Verify ROI config is loaded:

- [ ] **On startup:** Console shows "✓ ROI config loaded for <camera_id>"
- [ ] **Strict mode:** Console shows "Strict mode: True"
- [ ] **Allowed classes:** Console shows "Allowed classes: ['pine', 'worker']"
- [ ] **Forbidden zones:** Console shows "Forbidden zones: 1" (or 2 for cam_nave3_hornos_b_cam3)

### Statistics

At the end of the demo (after pressing Q), verify final stats show:

- [ ] **Total Alerts:** Alert count displayed
- [ ] **Strict Mode:** "Enabled" shown
- [ ] **Allowed Classes:** "pine, worker" listed

---

## 🔍 Detailed Test Scenarios

### Scenario 1: Pine Detection (No Alert)

**Setup:** Pine visible in video, inside ROI zone

**Expected:**
- Green bounding box
- Label: "pine 0.XX" (no "⚠ ALERT!" prefix)
- Alert count does NOT increase
- Box thickness: normal (3px)

**Validates:** Allowed classes don't trigger alerts in strict mode

---

### Scenario 2: Object Detection (Alert)

**Setup:** Object (tire, log, debris) visible in video, inside ROI zone

**Expected:**
- Red bounding box
- Label: "⚠ ALERT! object 0.XX"
- Alert count INCREASES
- Box thickness: thicker (4px)

**Validates:** Non-allowed classes trigger alerts in ROI zones

---

### Scenario 3: Unknown Object Detection (Alert)

**Setup:** Low-confidence detection (confidence < 0.15) inside ROI zone

**Expected:**
- Red bounding box
- Label: "⚠ ALERT! unknown_<class> 0.XX"
- Alert count INCREASES
- Detection flagged as `is_unknown: true`

**Validates:** Unknown/low-confidence detections trigger alerts

---

### Scenario 4: Detection Outside ROI (No Alert)

**Setup:** Object detected OUTSIDE forbidden ROI zone

**Expected:**
- Normal colored box (not red)
- No "⚠ ALERT!" prefix
- Alert count does NOT increase

**Validates:** ROI boundary detection works correctly

---

### Scenario 5: Multiple ROI Zones

**Setup:** Test cam_nave3_hornos_b_cam3 (has 2 ROI zones)

**Expected:**
- 2 yellow polygons displayed
- Detections inside EITHER zone trigger alerts
- Each zone labeled separately

**Validates:** Multiple ROI zones per camera work correctly

---

## 🐛 Common Issues & Solutions

### Issue: ROI polygons not showing

**Symptom:** No yellow polygons visible in demo

**Solutions:**
1. Verify `--roi-config` and `--camera-id` arguments provided
2. Check camera_id matches exactly in configs/rois.yaml
3. Verify ROI points are within frame dimensions

---

### Issue: All detections showing as alerts

**Symptom:** Even pines/workers show red boxes with "⚠ ALERT!"

**Solutions:**
1. Check `strict_mode: true` is set in configs/rois.yaml
2. Verify `allowed_classes: [pine, worker]` is configured
3. Check class names match exactly (case-sensitive)

---

### Issue: No alerts triggering

**Symptom:** Objects in ROI don't trigger alerts

**Solutions:**
1. Verify object detection centers are inside ROI polygon
2. Check ROI points form a valid closed polygon (min 3 points)
3. Ensure strict_mode is enabled

---

### Issue: Demo crashes on startup

**Symptom:** Error loading ROI config

**Solutions:**
1. Verify configs/rois.yaml exists and is valid YAML
2. Check camera_id exists in config file
3. Ensure all ROI points are valid integers
4. Run: `python -c "import yaml; yaml.safe_load(open('configs/rois.yaml'))"`

---

## 📊 Expected Test Results

### Per-Camera Validation

Test all 8 cameras to ensure consistency:

| Camera ID | ROI Zones | Expected Behavior |
|-----------|-----------|-------------------|
| cam_nave3_hornos_a_cam3 | 1 | Alerts on objects/unknown in ROI |
| cam_nave3_hornos_a | 1 | Alerts on objects/unknown in ROI |
| cam_nave3_hornos_b | 1 | Alerts on objects/unknown in ROI |
| cam_nave3_hornos_b_cam3 | **2** | Alerts in BOTH ROI zones |
| cam_nave4_difusor_a | 1 | Alerts on objects/unknown in ROI |
| cam_nave4_difusor_a_cam3 | 1 | Alerts on objects/unknown in ROI |
| cam_nave4_difusor_b | 1 | Alerts on objects/unknown in ROI |
| cam_nave4_difusor_b_cam3 | 1 | Alerts on objects/unknown in ROI |

---

## ✅ Success Criteria

Phase 1 is validated when:

- [x] ROI zones display correctly for all 8 cameras
- [x] Whitelist logic works (pine/worker allowed, everything else alerts)
- [x] Visual indicators clearly show alerts (red boxes, thick borders, labels)
- [x] Alert counts track correctly
- [x] Unknown objects are detected and flagged
- [x] Multiple ROI zones per camera work
- [x] Configuration loaded successfully
- [x] No crashes or errors during demo

---

## 🎉 Phase 1 Complete!

Once all validation items are checked, Phase 1 is complete and production-ready:

✅ **Whitelist alert logic** - Implemented and tested
✅ **Unknown object detection** - Working correctly
✅ **Enhanced alert events** - All fields populated
✅ **ROI configuration** - All 8 cameras configured
✅ **Visual demo** - Alerts clearly visible

**Next Step:** Proceed to Phase 2 (ML-Focused API Architecture)

---

## 📝 Test Log Template

Use this template to record your validation:

```
Date: _______________
Tester: _______________

Camera: cam_nave3_hornos_b_cam3
✓ ROI zones displayed: YES / NO
✓ Alert logic working: YES / NO
✓ Visual indicators correct: YES / NO
✓ Alert count tracking: YES / NO
Issues found: _______________________

Camera: cam_nave3_hornos_a_cam3
✓ ROI zones displayed: YES / NO
✓ Alert logic working: YES / NO
✓ Visual indicators correct: YES / NO
✓ Alert count tracking: YES / NO
Issues found: _______________________

[... repeat for each camera ...]

Overall Result: PASS / FAIL
Notes: _______________________________
```
