#!/usr/bin/env python3
"""
Extract and de-duplicate frames from raw videos into data/frames.

Flow alignment (videos -> frames -> tiles_pool -> tiles_man -> rounds):
- reads videos under data/videos (recursively)
- samples frames at SAMPLE_FPS
- keeps a frame only if its mean absolute difference vs. last kept frame exceeds DIFF_THRESHOLD
- writes frames to data/frames/<video_stem>/frame_{:06d}.jpg
- writes a JSON manifest describing kept frames (frames_manifest.json)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------
VIDEOS_ROOT = Path("data/videos")
FRAMES_ROOT = Path("data/frames")
MANIFEST_JSON = FRAMES_ROOT / "frames_manifest.json"

SAMPLE_FPS = 1  # frames per second to sample
DIFF_THRESHOLD = 2.0  # mean absolute difference threshold (0-255 scale)
# ----------------------------------------


def process_video(video_path: Path, out_dir: Path, sample_fps: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(int(round(native_fps / sample_fps)) if native_fps > 0 else 1, 1)

    kept = []
    frame_idx = 0
    kept_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_val = None
        if prev_gray is None:
            keep = True
        else:
            diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            diff_val = float(diff.mean())
            keep = diff_val >= DIFF_THRESHOLD

        if keep:
            filename = f"frame_{kept_idx:06d}.jpg"
            out_path = out_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(out_path), frame)
            if success:
                kept.append(
                    {
                        "video": str(video_path.relative_to(VIDEOS_ROOT)),
                        "frame_filename": filename,
                        "frame_path": str(out_path.relative_to(FRAMES_ROOT)),
                        "frame_index_in_video": frame_idx,
                        "kept_index": kept_idx,
                        "sample_fps": sample_fps,
                        "native_fps": native_fps,
                        "mad_prev": diff_val,
                    }
                )
            prev_gray = gray
            kept_idx += 1
        frame_idx += 1

    cap.release()
    return kept


def main():
    if not VIDEOS_ROOT.exists():
        raise FileNotFoundError(f"Videos root not found: {VIDEOS_ROOT}")

    all_entries = []
    for video_path in VIDEOS_ROOT.rglob("*"):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue

        out_dir = FRAMES_ROOT / video_path.stem
        entries = process_video(video_path, out_dir, SAMPLE_FPS)
        all_entries.extend(entries)
        print(f"[INFO] {video_path}: kept {len(entries)} frames")

    manifest = {
        "videos_root": str(VIDEOS_ROOT),
        "frames_root": str(FRAMES_ROOT),
        "sample_fps": SAMPLE_FPS,
        "diff_threshold": DIFF_THRESHOLD,
        "total_kept": len(all_entries),
        "files": all_entries,
    }

    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Manifest written to {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
