#!/usr/bin/env python3
"""
Split tiles_man into 4 balanced, non-overlapping rounds using JSON metadata.

Inputs:
  - data/tiles_pool/tiles_man/images/*.jpg
  - data/tiles_pool/tiles_man/metadata_man.json

Outputs:
  - data/tiles_pool/tiles_man/tiles_round1/images/*.jpg
  - data/tiles_pool/tiles_man/tiles_round1/metadata_round1.json
  - data/tiles_pool/tiles_man/tiles_round2/images/*.jpg
  - data/tiles_pool/tiles_man/tiles_round2/metadata_round2.json
  - data/tiles_pool/tiles_man/tiles_round3/images/*.jpg
  - data/tiles_pool/tiles_man/tiles_round3/metadata_round3.json
  - data/tiles_pool/tiles_man/tiles_round4/images/*.jpg
  - data/tiles_pool/tiles_man/tiles_round4/metadata_round4.json
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# ---------------- CONFIG ----------------
MAN_IMAGES_DIR = Path("data/tiles_pool/tiles_man/images")
MAN_META_JSON = Path("data/tiles_pool/tiles_man/metadata_man.json")

ROUNDS = [
    {
        "name": "round1",
        "images_dir": Path("data/tiles_pool/tiles_man/tiles_round1/images"),
        "meta_json": Path("data/tiles_pool/tiles_man/tiles_round1/metadata_round1.json"),
    },
    {
        "name": "round2",
        "images_dir": Path("data/tiles_pool/tiles_man/tiles_round2/images"),
        "meta_json": Path("data/tiles_pool/tiles_man/tiles_round2/metadata_round2.json"),
    },
    {
        "name": "round3",
        "images_dir": Path("data/tiles_pool/tiles_man/tiles_round3/images"),
        "meta_json": Path("data/tiles_pool/tiles_man/tiles_round3/metadata_round3.json"),
    },
    {
        "name": "round4",
        "images_dir": Path("data/tiles_pool/tiles_man/tiles_round4/images"),
        "meta_json": Path("data/tiles_pool/tiles_man/tiles_round4/metadata_round4.json"),
    },
]

RANDOM_SEED = 42  # deterministic splitting
# ----------------------------------------


def load_rows() -> List[Dict]:
    if not MAN_META_JSON.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {MAN_META_JSON}")
    if not MAN_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {MAN_IMAGES_DIR}")

    with MAN_META_JSON.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = meta.get("files", [])
    if not rows:
        raise RuntimeError("No rows found in metadata_clean.json")

    # Normalize filename field
    for row in rows:
        tile_filename = row.get("tile_filename") or row.get("filename")
        if not tile_filename:
            raise KeyError("Missing tile filename in metadata row")
        row["tile_filename"] = tile_filename
    return rows


def write_round_metadata(meta_path: Path, images_dir: Path, rows: List[Dict]):
    by_extension = defaultdict(int)
    by_folder = defaultdict(int)
    dimensions_hist = defaultdict(int)
    files = []

    for row in rows:
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
                "size_bytes": (images_dir / tile_filename).stat().st_size
                if (images_dir / tile_filename).exists()
                else None,
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
        "root": str(images_dir),
        "summary": {
            "total_images": len(rows),
            "by_extension": dict(by_extension),
            "by_folder": dict(by_folder),
            "dimensions_hist": dict(dimensions_hist),
        },
        "files": files,
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as jf:
        json.dump(meta_json, jf, indent=2)


def main():
    rows = load_rows()
    total_tiles = len(rows)
    print(f"Total tiles in tiles_man: {total_tiles}")

    # Group rows by folder
    by_folder: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        folder = row.get("folder", "")
        by_folder[folder].append(row)

    # Prepare data structures to collect rows per round
    round_rows = {r["name"]: [] for r in ROUNDS}

    rng = random.Random(RANDOM_SEED)

    # Split each folder across 4 rounds
    for folder, folder_rows in by_folder.items():
        n = len(folder_rows)
        print(f"Folder '{folder}': {n} tiles")

        # Shuffle in-place for this folder
        rng.shuffle(folder_rows)

        base = n // 4
        rem = n % 4

        counts_per_round = [base + (1 if i < rem else 0) for i in range(4)]
        assert sum(counts_per_round) == n, "Bug in round partitioning logic"

        idx = 0
        for round_idx, round_def in enumerate(ROUNDS):
            count = counts_per_round[round_idx]
            if count == 0:
                continue
            slice_rows = folder_rows[idx : idx + count]
            idx += count
            round_rows[round_def["name"]].extend(slice_rows)

    # Create output dirs and write per-round results
    for round_def in ROUNDS:
        round_name = round_def["name"]
        images_dir = round_def["images_dir"]
        meta_json = round_def["meta_json"]

        images_dir.mkdir(parents=True, exist_ok=True)
        rows_list = round_rows[round_name]
        print(f"{round_name}: {len(rows_list)} tiles")

        if not rows_list:
            print(f"Warning: {round_name} has no tiles.")
            continue

        # Copy images
        for row in rows_list:
            tile_filename = row["tile_filename"]
            src = MAN_IMAGES_DIR / tile_filename
            if not src.exists():
                continue
            dst = images_dir / tile_filename
            shutil.copy2(src, dst)

        # Write JSON metadata
        write_round_metadata(meta_json, images_dir, rows_list)

    total_assigned = sum(len(v) for v in round_rows.values())
    print(f"Total assigned tiles across all rounds: {total_assigned}")
    if total_assigned != total_tiles:
        print("WARNING: total assigned != total tiles! Check logic.")
    else:
        print("All tiles assigned exactly once.")


if __name__ == "__main__":
    main()
