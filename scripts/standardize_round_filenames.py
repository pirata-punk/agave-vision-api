#!/usr/bin/env python3
"""
Standardize image/label filenames in tiles_round directories (round1-4).

Handles patterns seen across rounds:
1) URL-encoded stems (round1): decode %XX sequences.
2) Hashed prefixes/double underscores (round2): strip leading hex/dash tokens.
3) images\ prefixes/pseudo paths (round3/4): strip leading images/, replace slashes.
4) Global: remove spaces by converting them to underscores; collapse multiple underscores.

Behavior:
- Applies both decode and prefix/underscore normalization to stems for images and labels.
- Renames files in-place when the normalized name differs and the target does not already exist.
- Prints a summary and unmatched stems after renaming to help verify 1:1 mapping.

Safety:
- Skips a rename if the target filename already exists (prints a warning).
- Default ROUND_DIR targets tiles_round1; adjust before running on other rounds.
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path
from typing import Iterable, Set, List

# ---------------- CONFIG ----------------
ROUND_DIRS: List[Path] = [
    Path("data/tiles_pool/tiles_man/tiles_round1"),
    Path("data/tiles_pool/tiles_man/tiles_round2"),
    Path("data/tiles_pool/tiles_man/tiles_round3"),
    Path("data/tiles_pool/tiles_man/tiles_round4"),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".txt"
# ----------------------------------------


def decode_stem(stem: str) -> str:
    return urllib.parse.unquote(stem)


def strip_hashed_prefix(stem: str) -> str:
    """
    Remove leading hex-ish token plus dash or double underscore, e.g.,
    "0a8f977d-NAME" -> "NAME", "0a5b0597__NAME" -> "NAME".
    """
    import re

    m = re.match(r"^[0-9a-fA-F]{6,}[-_]{1,2}(.+)$", stem)
    return m.group(1) if m else stem


def collapse_underscores(stem: str) -> str:
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem


def normalize_stem(stem: str) -> str:
    import re

    # decode %xx
    stem = decode_stem(stem)
    # strip hashed prefix
    stem = strip_hashed_prefix(stem)
    # drop leading images\ or images/ artifacts (and variants)
    stem = re.sub(r"^_?images[\\/]+", "", stem)
    stem = re.sub(r"^__?images[\\/]+", "", stem)
    # replace remaining backslashes/slashes with underscores to avoid path issues
    stem = stem.replace("\\", "_").replace("/", "_")
    # collapse multiple underscores
    stem = collapse_underscores(stem)
    # replace spaces with underscores
    stem = stem.replace(" ", "_")
    # collapse again in case replacements introduced doubles
    stem = collapse_underscores(stem)
    # strip stray leading/trailing underscores
    stem = stem.strip("_")
    return stem


def rename_with_normalize(paths: Iterable[Path], allowed_exts: Set[str]) -> int:
    renamed = 0
    for path in paths:
        if path.suffix.lower() not in allowed_exts:
            continue
        norm_stem = normalize_stem(path.stem)
        if norm_stem == path.stem:
            continue
        target = path.with_name(norm_stem + path.suffix)
        if target.exists():
            print(f"[WARN] Target already exists, skipping rename: {target}")
            continue
        print(f"[INFO] Renaming {path.name} -> {target.name}")
        path.rename(target)
        renamed += 1
    return renamed


def collect_stems(dir_path: Path, exts: Set[str]) -> Set[str]:
    stems = set()
    for p in dir_path.iterdir():
        if p.suffix.lower() in exts:
            stems.add(p.stem)
    return stems


def main():
    for round_dir in ROUND_DIRS:
        images_dir = round_dir / "images"
        labels_dir = round_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"[ERROR] Missing images or labels directory under {round_dir}")
            continue

        print(f"[INFO] Processing {round_dir}")
        img_renamed = rename_with_normalize(images_dir.iterdir(), IMAGE_EXTS)
        lbl_renamed = rename_with_normalize(labels_dir.iterdir(), {LABEL_EXT})

        img_stems = collect_stems(images_dir, IMAGE_EXTS)
        lbl_stems = collect_stems(labels_dir, {LABEL_EXT})

        missing_labels = sorted(img_stems - lbl_stems)
        missing_images = sorted(lbl_stems - img_stems)

        print(f"[DONE] {round_dir.name}: Renamed images: {img_renamed}, labels: {lbl_renamed}")
        print(f"[SUMMARY] {round_dir.name}: Images: {len(img_stems)}, Labels: {len(lbl_stems)}")
        print(f"[SUMMARY] {round_dir.name}: Images without labels: {len(missing_labels)}")
        if missing_labels:
            print("  Examples:", missing_labels[:10])
        print(f"[SUMMARY] {round_dir.name}: Labels without images: {len(missing_images)}")
        if missing_images:
            print("  Examples:", missing_images[:10])


if __name__ == "__main__":
    main()
