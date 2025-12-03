"""
Filter tiles_pool into a balanced manual-label set (tiles_man) using edge content.

Flow alignment (videos -> frames -> tiles_pool -> tiles_man -> rounds):
- consumes tiles_pool/metadata.json
- drops low-information tiles via edge threshold
- applies per-folder quotas and global cap
- writes images to tiles_man/images and metadata to tiles_man/metadata_man.json (JSON only)
"""

import os
import random
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
# Input: tiles pool
TILES_DIR = Path("tiles_pool/images")
META_JSON = Path("tiles_pool/metadata.json")

# Output: manually labeled candidate set (do not overwrite existing labeled data)
OUT_DIR = Path("tiles_man/images")
OUT_META_JSON = Path("tiles_man/metadata_man.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum edge content required to keep a tile
EDGE_THRESHOLD = 1500  # tune 1200–2000 depending on dataset

# Max tiles per folder (balancing cameras)
FOLDER_QUOTAS = {
    "NAVE 4_DIFUSOR  B_20250923180631_20250923180839": 350,
    "NAVE 4_DIFUSOR  B_20250923171010_20250923172014": 250,
    "NAVE 4_DIFUSOR  B_20250923192100_20250923193308": 250,
    "NAVE 3_ HORNOS B_20250923125053_20250923131505": 350,
    "NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506": 250,
    "NAVE 3_HORNOS A CAM 3_20250923162408_20250923165118": 250,
    # small folders: no quota
}

# Global cap on how many tiles we keep in total
MAX_TOTAL_TILES = 2000
# ------------------------------------------------------


def compute_edge_strength(img):
    """Return edge pixel count using Canny."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 140)
    return int(np.sum(edges > 0))


def main():
    # 0. Read metadata
    if not META_JSON.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {META_JSON}")
    if not TILES_DIR.exists():
        raise FileNotFoundError(f"Tiles directory not found: {TILES_DIR}")

    with META_JSON.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta.get("files", [])

    # 1. Edge-based filtering
    kept = []
    for row in rows:
        tile_filename = row.get("filename") or row.get("tile_filename")
        if not tile_filename:
            continue
        tile_path = TILES_DIR / tile_filename
        if not tile_path.exists():
            continue

        img = cv2.imread(str(tile_path))
        if img is None:
            continue

        edges = compute_edge_strength(img)
        if edges < EDGE_THRESHOLD:
            # too few edges -> likely empty / non-pina tile
            continue

        row["edge_strength"] = edges
        row["tile_filename"] = tile_filename
        kept.append(row)

    print(f"After edge filtering: {len(kept)} tiles")

    # 2. Folder balancing based on FOLDER_QUOTAS
    folder_counts = defaultdict(int)
    balanced = []

    # shuffle before applying quotas to avoid bias
    random.shuffle(kept)

    for row in kept:
        folder = row.get("folder", "")
        quota = FOLDER_QUOTAS.get(folder, None)

        if quota is None:
            # no quota -> keep all (subject to global cap later)
            balanced.append(row)
        else:
            if folder_counts[folder] < quota:
                folder_counts[folder] += 1
                balanced.append(row)

    print(f"After folder balancing: {len(balanced)} tiles")

    # 3. Global random sampling
    if len(balanced) > MAX_TOTAL_TILES:
        random.shuffle(balanced)
        balanced = balanced[:MAX_TOTAL_TILES]

    print(f"Final tile count: {len(balanced)}")

    # 4. Copy tiles + write new metadata
    out_rows = []
    for row in balanced:
        tile_filename = row["tile_filename"]
        src = TILES_DIR / tile_filename
        if not src.exists():
            continue

        dst = OUT_DIR / tile_filename
        shutil.copy2(src, dst)
        out_rows.append(row)

    if out_rows:
        OUT_META_JSON.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON metadata similar to scan_frames_metadata.py
        by_extension = defaultdict(int)
        by_folder = defaultdict(int)
        dimensions_hist = defaultdict(int)
        files = []

        for row in out_rows:
            tile_filename = row["tile_filename"]
            ext = Path(tile_filename).suffix.lower()
            by_extension[ext] += 1
            by_folder[row.get("folder", "")] += 1
            dimensions_hist[f"{row['width']}x{row['height']}"] += 1

            files.append(
                {
                    "relative_path": tile_filename,
                    "folder": row.get("folder", ""),
                    "filename": tile_filename,
                    "extension": ext,
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "size_bytes": (OUT_DIR / tile_filename).stat().st_size,
                    "source_relpath": row.get("source_relpath", ""),
                    "x0": int(row["x0"]),
                    "y0": int(row["y0"]),
                    "orig_width": int(row["orig_width"]),
                    "orig_height": int(row["orig_height"]),
                    "edge_strength": int(row.get("edge_strength", 0)),
                    "tile_id": int(row["tile_id"]),
                }
            )

        meta_json = {
            "root": str(OUT_DIR),
            "summary": {
                "total_images": len(out_rows),
                "by_extension": dict(by_extension),
                "by_folder": dict(by_folder),
                "dimensions_hist": dict(dimensions_hist),
            },
            "files": files,
        }

        with OUT_META_JSON.open("w", encoding="utf-8") as jf:
            json.dump(meta_json, jf, indent=2)

    print(f"Saved cleaned tiles in: {OUT_DIR}")
    print(f"Saved cleaned metadata in: {OUT_META_JSON}")


if __name__ == "__main__":
    main()
