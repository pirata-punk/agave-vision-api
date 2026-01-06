Here’s an updated version of the smart_tile_selection.md idea, now grounded on your actual dataset stats from frames_metadata.json. ￼

You can just overwrite your existing doc with this if you like.

⸻

Smart Tile Selection for Agave Piña Detection

This document describes how we’ll generate smart tiles from the frames in data/frames and prepare them for labeling + YOLOv8n training. It’s tailored to the current dataset snapshot.

⸻

0. Dataset snapshot (from frames_metadata.json)

From the metadata scan we know: ￼
• Total frames: 2 501 (all .jpg)
• By folder (roughly per camera / recording):
• NAVE 4*DIFUSOR B_20250923171010_20250923172014: 898
• NAVE 4_DIFUSOR B_20250923180631_20250923180839: 192
• NAVE 4_DIFUSOR B_20250923192100_20250923193308: 899
• NAVE 3* HORNOS B*20250923125053_20250923131505: 347
• NAVE 3_HORNOS A CAM 3_20250923162408_20250923165118: 70
• NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506: 69
• NAVE 4_DIFUSOR A_20250929112706_20250929113002: 18
• NAVE 4_DIFUSOR A_20250929123952_20250929124208: 4
• NAVE 3* HORNOS A_20250923162907_20250923165519: 4
• Resolutions:
• 2560×1440: 2 475 images (≈ 99%)
• 2688×1520: 26 images

So:
• DIFUSOR B dominates the dataset (~80% of frames).
• Almost everything is 2560×1440, so we can optimize the tiling around that resolution while still supporting others.

⸻

1. Design decisions

1.1 Tile size & overlap

Given 2560×1440 frames:
• Tile size: 640×640
• Overlap: 128 px (20%) horizontally and vertically
• YOLO imgsz: 640

Why this works well for 2560×1440:
• Horizontal: step = 640 − 128 = 512
• Positions ≈ x ∈ {0, 512, 1024, 1536, 1920} → 5 tiles
• Vertical: same step
• For 1440 height: positions ≈ y ∈ {0, 512, 800} → 3 tiles
• Total ≈ 15 tiles per 2560×1440 frame before filtering.

For the few 2688×1520 frames, the same logic still works; the script computes windows from w, h dynamically.

If 15 tiles per frame is too many, we’ll downsample tiles later (Section 3).

1.2. Edge visibility rule

Keep the previous consistent rule:
• If ≥ 30% of a piña/person/object is visible in a tile → label it.
• If you wouldn’t count it in real life (tiny sliver at the edge) → skip.
• It’s fine if the same physical piña appears in overlapping tiles; YOLO treats them as independent examples.

⸻

2. Choose base frames per camera (using folder stats)

We don’t want all 2 501 frames in the first labeling iteration, especially since DIFUSOR B is heavily over-represented.

Goal: roughly balance cameras in the labeled set.

Suggested strategy: 1. Decide a target number of frames per folder for Round 1, e.g.:
• 80–100 frames from each large DIFUSOR B folder.
• All frames from small folders (e.g. the ones with ≤70 images). 2. Sample uniformly within each folder (you already have a cosine-distance / sharpness selector – reuse it inside each folder).

Example Round-1 plan (just illustrative):
• NAVE 4*DIFUSOR B_20250923171010_20250923172014: sample 100
• NAVE 4_DIFUSOR B_20250923180631_20250923180839: keep all 192
• NAVE 4_DIFUSOR B_20250923192100_20250923193308: sample 100
• NAVE 3* HORNOS B*20250923125053_20250923131505: sample 100
• NAVE 3_HORNOS A CAM 3*...: keep all 70
• NAVE 3*HORNOS B CAM 3*...: keep all 69
• NAVE 4*DIFUSOR A*_: keep all 22
• NAVE 3* HORNOS A*_: keep all 4

That gives you on the order of 650–700 base frames, still manageable and better balanced.

⸻

3. Tiling script adapted to data/frames/

3.1. Key changes compared to the earlier doc
• Use FRAMES_ROOT = "data/frames" and walk recursively, because your frames are nested in camera-specific folders.
• Optional: integrate frames_metadata.json to subsample frames per folder instead of purely filename-based filtering.
• Update overlap to 128 instead of 64.

3.2. Example script

import os
import cv2
import csv
from pathlib import Path
from collections import defaultdict

# ------------ CONFIG ------------

FRAMES_ROOT = Path("data/frames") # where all frames live
OUT_DIR = Path("tiles/images") # where tiles will be written
META_CSV = Path("tiles/metadata.csv") # tile-level metadata

TILE_SIZE = 640
OVERLAP = 128 # pixels
MIN_TILE_STD = 3.0 # skip almost empty/flat tiles (optional)

# Optional camera balancing: max frames per folder

MAX_FRAMES_PER_FOLDER = 120 # None = no limit

# --------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def iter_images(root: Path):
"""Yield (path, folder_name) for all image files under root."""
for p in root.rglob("\*.jpg"): # top-level folder under data/frames
try:
folder_name = p.relative_to(root).parts[0]
except IndexError:
folder_name = ""
yield p, folder_name

def sliding_windows(w, h, tile_size, overlap):
step = tile_size - overlap

    xs, ys = [], []
    x = 0
    while True:
        if x + tile_size >= w:
            xs.append(max(0, w - tile_size))
            break
        xs.append(x)
        x += step

    y = 0
    while True:
        if y + tile_size >= h:
            ys.append(max(0, h - tile_size))
            break
        ys.append(y)
        y += step

    for y0 in ys:
        for x0 in xs:
            yield x0, y0

def main():
rows = []
tile_id = 0

    # Optional per-folder cap
    folder_counts = defaultdict(int)

    for img_path, folder_name in iter_images(FRAMES_ROOT):
        if MAX_FRAMES_PER_FOLDER is not None:
            if folder_counts[folder_name] >= MAX_FRAMES_PER_FOLDER:
                continue
            folder_counts[folder_name] += 1

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] could not read {img_path}")
            continue

        h, w = img.shape[:2]
        base_name = img_path.stem  # e.g. frame_000100
        base_prefix = f"{folder_name}_{base_name}"

        for x0, y0 in sliding_windows(w, h, TILE_SIZE, OVERLAP):
            x1 = x0 + TILE_SIZE
            y1 = y0 + TILE_SIZE
            tile = img[y0:y1, x0:x1]

            if tile.std() < MIN_TILE_STD:
                continue

            tile_filename = f"{base_prefix}_x{x0}_y{y0}.jpg"
            tile_path = OUT_DIR / tile_filename
            cv2.imwrite(str(tile_path), tile)

            rows.append({
                "tile_id": tile_id,
                "tile_filename": tile_filename,
                "source_image": img_path.name,
                "source_relpath": str(img_path.relative_to(FRAMES_ROOT)),
                "folder": folder_name,
                "x0": x0,
                "y0": y0,
                "width": TILE_SIZE,
                "height": TILE_SIZE,
                "orig_width": w,
                "orig_height": h,
            })
            tile_id += 1

    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with META_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Created {len(rows)} tiles.")

if **name** == "**main**":
main()

Notes:
• Tiles get names like
NAVE_4_DIFUSOR\_\_B_20250923171010_20250923172014_frame_000100_x0_y0.jpg
so you can trace back to both folder and original frame.
• MAX_FRAMES_PER_FOLDER lets you cap contribution from over-represented cameras in the first iteration. You can always rerun with a higher cap later.

⸻

4. Estimating tile counts & labeling load

With 640×640 tiles, overlap 128:
• Per 2560×1440 frame: ~15 tiles
• If you sampled ~700 frames for Round-1 labeling:
• Raw tiles: 700 × 15 ≈ 10 500
• After MIN_TILE_STD filtering & heuristics (e.g., removing almost-empty tiles), you might end up around 6–8k tiles.
• You can then subsample tiles again before labeling (e.g., randomly choose 2–3k for first iteration).

Recommendation: 1. Run the tiler on your Round-1 frame subset. 2. Inspect tiles/metadata.csv:
• Check tile counts per folder.
• Optionally sample a fixed number of tiles per folder (e.g. 400–600) for the initial Label Studio project. 3. Keep the unused tiles around; they can be added in later training rounds.

⸻

5. Labeling guidelines (unchanged but reiterated)

We keep the existing Label Studio config and annotation rules from the previous document, since they’re independent of resolution: ￼
• Classes
• pina
• worker
• object
• General rules
• Separate box per object.
• Boxes reasonably tight; some background is fine.
• Apply the ≥ 30% visible rule at tile borders.
• Piñas: label all visually separable piñas; skip tiny slivers.
• Persons: label any worker with recognizable PPE; skip tiny fragments.
• Foreign objects: label large/log-shaped / tire / debris chunks; ignore dust and tiny fragments.

⸻

6. Train/val/test splitting at tile-group level

No change in logic from the earlier doc, but note:
• Use source_relpath (or source_image + folder) from tiles/metadata.csv as the group key so tiles from the same original frame all go to the same split. ￼

The example train_test_split code from the original smart_tile_selection.md still applies; just update the group column if you adopt source_relpath.

⸻

7. Summary of what actually changed vs original doc
   1. Linked to actual dataset stats (2 501 frames, 9 folders, 2 resolutions).
   2. Tile parameters tuned for 2560×1440 (still 640×640, but OVERLAP=128 and explicit expected tile counts).
   3. Tiling script updated to:
      • Walk data/frames recursively.
      • Track folder and source_relpath.
      • Optionally cap frames per folder to balance cameras.
   4. Left the Label Studio, YOLO conversion, and train/val/test sections conceptually the same; they remain valid for this dataset.
