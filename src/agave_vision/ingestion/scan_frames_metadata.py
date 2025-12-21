"""
Scan data/frames and collect metadata for all images.

Outputs: data/frames/frames_metadata.json
"""

import json
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

from PIL import Image


# ---- CONFIG ----
FRAMES_ROOT = Path("data/frames")
OUTPUT_JSON = FRAMES_ROOT / "frames_metadata.json"

# Which file extensions to treat as images
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ---------------


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def scan_frames(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Frames root does not exist: {root}")

    files_metadata = []
    by_extension = Counter()
    by_folder = Counter()
    dimensions_hist = Counter()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not is_image(path):
            continue

        # Read basic file info
        rel_path = path.relative_to(root)
        size_bytes = path.stat().st_size
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        # Determine top-level folder name (if any)
        parts = rel_path.parts
        if len(parts) > 1:
            top_folder = parts[0]
        else:
            top_folder = ""  # image directly under data/frames

        # Open image for metadata
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format  # e.g. "JPEG"
        except Exception as e:
            print(f"[WARN] Failed to open {rel_path}: {e}")
            width = height = None
            mode = None
            fmt = None

        ext = path.suffix.lower()

        file_meta = {
            "relative_path": str(rel_path).replace(os.sep, "/"),
            "folder": top_folder,
            "filename": path.name,
            "extension": ext,
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
            "size_bytes": size_bytes,
            "mtime": mtime,
        }
        files_metadata.append(file_meta)

        # Update aggregates if we actually got a size
        by_extension[ext] += 1
        by_folder[top_folder] += 1
        if width is not None and height is not None:
            dimensions_hist[f"{width}x{height}"] += 1

    summary = {
        "total_images": len(files_metadata),
        "by_extension": dict(by_extension),
        "by_folder": dict(by_folder),
        "dimensions_hist": dict(dimensions_hist),
    }

    result = {
        "root": str(root),
        "summary": summary,
        "files": files_metadata,
    }

    return result


def main():
    print(f"Scanning frames under: {FRAMES_ROOT}")
    metadata = scan_frames(FRAMES_ROOT)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to: {OUTPUT_JSON}")
    print(f"Total images: {metadata['summary']['total_images']}")


if __name__ == "__main__":
    main()
