"""
Generate 640x640 tiles with 20% overlap from the curated frame subset.

Flow alignment (videos -> frames -> tiles_pool -> tiles_man -> rounds):
- This script consumes deduped frames under data/frames (plus selected_frames.json order if present)
- Balances frame contribution per camera folder
- Writes tiles into tiles_pool/images and metadata to tiles_pool/metadata.json (JSON only)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2

# ---------------- CONFIG ---------------- #
FRAMES_ROOT = Path("data/frames")
SELECTED_FRAMES_JSON = FRAMES_ROOT / "selected_frames.json"

# Tiles pool output (do not overwrite existing labeled sets; adjust paths if needed)
OUT_DIR = Path("data/tiles_pool/images")
META_JSON = Path("data/tiles_pool/metadata.json")

TILE_SIZE = 640
OVERLAP = 128  # 20%
MIN_TILE_STD = 3.0  # skip almost-uniform tiles

# Folder-level caps for Round-1 balancing (None = keep all frames)
PER_FOLDER_LIMITS: Dict[str, Optional[int]] = {
    # Sample approx. 100 frames from each large DIFUSOR B sequence
    "NAVE 4_DIFUSOR  B_20250923171010_20250923172014": 100,
    "NAVE 4_DIFUSOR  B_20250923192100_20250923193308": 100,
    # Keep all frames for the medium DIFUSOR B sequence (already small)
    "NAVE 4_DIFUSOR  B_20250923180631_20250923180839": None,
    # Sample 100 from the larger Hornos view
    "NAVE 3_ HORNOS B_20250923125053_20250923131505": 100,
    # Other folders fall back to DEFAULT_MAX_FRAMES (None = keep all)
}
DEFAULT_MAX_FRAMES_PER_FOLDER: Optional[int] = None

# File extensions treated as images when selected_frames.json is absent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ---------------------------------------- #


def load_candidate_frames() -> List[Path]:
    """
    Return an ordered list of frame paths to consider.
    selected_frames.json is preferred when present so we re-use the
    sharpness/embedding ranking produced earlier.
    """
    if SELECTED_FRAMES_JSON.exists():
        with open(SELECTED_FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Sort defensively by rank in case the json isn't already ordered
        data.sort(key=lambda row: row.get("rank", 0))
        frames: List[Path] = []
        for row in data:
            frame_path = Path(row["path"])
            if not frame_path.is_absolute():
                frame_path = Path.cwd() / frame_path
            if not frame_path.exists():
                continue
            frames.append(frame_path)
        return frames

    print("[INFO] selected_frames.json not found; walking FRAMES_ROOT directly.")
    frames = [
        p
        for p in FRAMES_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    frames.sort()
    return frames


def resolve_folder(frame_path: Path) -> str:
    rel = frame_path.relative_to(FRAMES_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def allowed_in_folder(folder: str, folder_counts: Dict[str, int]) -> bool:
    limit = PER_FOLDER_LIMITS.get(folder, DEFAULT_MAX_FRAMES_PER_FOLDER)
    if limit is None:
        return True
    return folder_counts[folder] < limit


def sliding_windows(
    width: int, height: int, tile: int, overlap: int
) -> Iterator[Tuple[int, int]]:
    step = tile - overlap
    xs: List[int] = []
    ys: List[int] = []

    x = 0
    while True:
        if x + tile >= width:
            xs.append(max(0, width - tile))
            break
        xs.append(x)
        x += step

    y = 0
    while True:
        if y + tile >= height:
            ys.append(max(0, height - tile))
            break
        ys.append(y)
        y += step

    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def main():
    if not FRAMES_ROOT.exists():
        raise SystemExit(f"Frames directory not found: {FRAMES_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_JSON.parent.mkdir(parents=True, exist_ok=True)

    candidate_frames = load_candidate_frames()
    folder_counts = defaultdict(int)
    rows: List[dict] = []
    tile_id = 0

    for frame_path in candidate_frames:
        try:
            rel = frame_path.relative_to(Path.cwd())
        except ValueError:
            rel = frame_path

        # Ensure the path is under FRAMES_ROOT
        try:
            rel_to_root = rel.relative_to(FRAMES_ROOT)
        except ValueError:
            # If selected_frames.json included absolute paths, convert now
            rel_to_root = frame_path.resolve().relative_to(FRAMES_ROOT.resolve())

        folder = rel_to_root.parts[0] if len(rel_to_root.parts) else ""
        if not allowed_in_folder(folder, folder_counts):
            continue
        folder_counts[folder] += 1

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        height, width = img.shape[:2]
        base_prefix = f"{folder}_{frame_path.stem}"

        for x0, y0 in sliding_windows(width, height, TILE_SIZE, OVERLAP):
            x1 = min(width, x0 + TILE_SIZE)
            y1 = min(height, y0 + TILE_SIZE)
            tile = img[y0:y1, x0:x1]

            if tile.size == 0 or tile.std() < MIN_TILE_STD:
                continue

            tile_filename = f"{base_prefix}_x{x0}_y{y0}.jpg"
            tile_path = OUT_DIR / tile_filename
            success = cv2.imwrite(str(tile_path), tile)
            if not success:
                print(f"[WARN] Failed to write tile {tile_path}")
                continue

            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_filename": tile_filename,
                    "tile_relpath": str(tile_path.relative_to(OUT_DIR)).replace(
                        "\\", "/"
                    ),
                    "source_relpath": str(rel_to_root).replace("\\", "/"),
                    "folder": folder,
                    "x0": x0,
                    "y0": y0,
                    "width": tile.shape[1],
                    "height": tile.shape[0],
                    "orig_width": width,
                    "orig_height": height,
                }
            )
            tile_id += 1

    if rows:
        # Emit JSON metadata similar to scan_frames_metadata.py
        by_extension = defaultdict(int)
        by_folder = defaultdict(int)
        dimensions_hist = defaultdict(int)
        files = []

        for row in rows:
            # tile_relpath is already relative to OUT_DIR
            rel_path = Path(row["tile_relpath"])
            ext = Path(row["tile_filename"]).suffix.lower()

            by_extension[ext] += 1
            by_folder[row["folder"]] += 1
            dimensions_hist[f"{row['width']}x{row['height']}"] += 1

            # File size on disk
            tile_abs = OUT_DIR / rel_path
            try:
                size_bytes = tile_abs.stat().st_size
            except FileNotFoundError:
                size_bytes = None

            files.append(
                {
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "folder": row["folder"],
                    "filename": row["tile_filename"],
                    "extension": ext,
                    "width": row["width"],
                    "height": row["height"],
                    "size_bytes": size_bytes,
                    "source_relpath": row["source_relpath"],
                    "x0": row["x0"],
                    "y0": row["y0"],
                    "orig_width": row["orig_width"],
                    "orig_height": row["orig_height"],
                    "tile_id": row["tile_id"],
                }
            )

        meta_json = {
            "root": str(OUT_DIR),
            "summary": {
                "total_images": len(rows),
                "by_extension": dict(by_extension),
                "by_folder": dict(by_folder),
                "dimensions_hist": dict(dimensions_hist),
            },
            "files": files,
        }

        with META_JSON.open("w", encoding="utf-8") as jf:
            json.dump(meta_json, jf, indent=2)

    print(f"[DONE] Created {len(rows)} tiles across {len(folder_counts)} folders.")
    for folder, count in sorted(folder_counts.items()):
        limit = PER_FOLDER_LIMITS.get(folder, DEFAULT_MAX_FRAMES_PER_FOLDER)
        lim_txt = "∞" if limit is None else str(limit)
        print(f"  {folder}: {count} frames used (limit {lim_txt})")


if __name__ == "__main__":
    main()
