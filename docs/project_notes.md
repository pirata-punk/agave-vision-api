# Agave Vision Project Notes

Use this doc as a running scratchpad for architecture, planning, and operational conventions.

## Pipeline Flow
- Videos → frames (dedup via MAD/SSIM) → tiles_pool → tiles_man → tiles_round{1..4} → tiles_yolo → train/infer.
- Scripts (no auto-run):
  - `scripts/extract_frames.py`: sample/dedup frames from `data/videos` → `data/frames` + `frames_manifest.json`.
  - `scripts/scan_frames_metadata.py`: summarize frames → `frames_metadata.json`.
  - `scripts/generate_tiles.py`: frames → `data/tiles_pool` tiles + metadata.json.
  - `scripts/clean_tiles.py`: tiles_pool → tiles_man (edge filter + quotas) → metadata_man.json.
  - `scripts/standardize_round_filenames.py`: normalize filenames in tiles_round1..4 (decode %xx, strip hashed prefixes, strip images/ artifacts, replace spaces with `_`, collapse `__`).
  - `scripts/split_rounds_from_clean.py`: (if needed) split tiles_man into rounds with JSON metadata.
  - `scripts/build_yolo_dataset.py`: unify rounds into `data/tiles_yolo` with train/val/test splits, `configs/yolo_data.yaml`, `metadata.json`.
  - `scripts/infer_alert.py`: sample alerting pipeline using ROIs + YOLO model.

## Filename Standardization Rules (rounds 1–4)
- Decode percent-encodings (`%20` → space) before normalization.
- Strip hashed prefixes like `0a8f977d-` or `0a5b0597__`.
- Remove leading `images/` or `images\` artifacts and replace any slashes/backslashes with `_`.
- Replace all spaces with `_`.
- Collapse multiple underscores (`__` → `_`) and trim stray leading/trailing `_`.
- Apply the same normalization to both images and labels to maintain 1:1 pairing.

## Dataset Construction
- Source: `data/tiles_pool/tiles_man/tiles_round{1..4}/images|labels`.
- Keep only pairs where both image and non-empty label exist.
- Deterministic split (seed=123) with ratios train 0.7 / val 0.15 / test 0.15.
- Output: `data/tiles_yolo/{images,labels}/{train,val,test}` plus `configs/yolo_data.yaml` and `data/tiles_yolo/metadata.json`.
- Class order (YOLO IDs): 0: object, 1: pine, 2: worker.

## Training Reference
- After dataset build: `yolo detect train model=ultralytics/yolov8n.pt data=configs/yolo_data.yaml epochs=100 imgsz=640 batch=16`.

## Alerting Reference
- Use YOLO detections; alert when class `object` enters forbidden ROI. Allow `pine` and `worker`.
- ROI config example at `configs/rois.example.yaml`.
- Sample code: `scripts/infer_alert.py`.

## Open Questions / TODO
- Validate remaining unmatched image/label stems after standardization; resolve any edge cases.
- Confirm final ROI polygons per camera.
- Hardware constraints for training/inference (batch sizing, model variant).
