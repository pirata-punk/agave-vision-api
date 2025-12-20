"""
Tile Generator

Generates fixed-size tiles from frames for YOLO training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from agave_vision.core.frames import sliding_window_tiles
from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class TileGenerator:
    """
    Generate tiles from frames using sliding window.

    Args:
        frames_dir: Directory containing extracted frames
        output_dir: Directory to save tiles
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles
        min_edge_ratio: Minimum ratio of image edges to keep tile

    Example:
        >>> generator = TileGenerator(
        ...     frames_dir="data/frames",
        ...     output_dir="data/tiles_pool"
        ... )
        >>> generator.generate_all_tiles()
    """

    def __init__(
        self,
        frames_dir: str | Path,
        output_dir: str | Path,
        tile_size: int = 640,
        overlap: int = 128,
        min_edge_ratio: float = 0.3,
    ):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_edge_ratio = min_edge_ratio

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_tiles(self) -> dict:
        """
        Generate tiles from all frames.

        Returns:
            Metadata dictionary with tile statistics
        """
        frame_files = list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png"))

        if not frame_files:
            logger.warning(f"No frame files found in {self.frames_dir}")
            return {}

        logger.info(f"Found {len(frame_files)} frames")

        total_tiles = 0
        metadata = {"frames_processed": 0, "total_tiles": 0, "tiles": []}

        for frame_file in tqdm(frame_files, desc="Generating tiles"):
            tiles_metadata = self.generate_tiles_from_frame(frame_file)
            total_tiles += len(tiles_metadata)
            metadata["frames_processed"] += 1
            metadata["tiles"].extend(tiles_metadata)

        metadata["total_tiles"] = total_tiles

        logger.info(
            f"Generated {total_tiles} tiles from {len(frame_files)} frames "
            f"(avg {total_tiles / len(frame_files):.1f} tiles/frame)"
        )

        # Save metadata
        self._save_metadata(metadata)

        return metadata

    def generate_tiles_from_frame(self, frame_path: Path) -> list[dict]:
        """
        Generate tiles from a single frame.

        Args:
            frame_path: Path to frame image

        Returns:
            List of tile metadata dictionaries
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Failed to read frame: {frame_path}")
            return []

        tiles_metadata = []

        for idx, (tile, x_offset, y_offset) in enumerate(
            sliding_window_tiles(frame, self.tile_size, self.overlap)
        ):
            # Check edge ratio (skip tiles with too much padding)
            actual_height, actual_width = tile.shape[:2]
            if actual_height < self.tile_size * self.min_edge_ratio or actual_width < self.tile_size * self.min_edge_ratio:
                continue

            # Save tile
            tile_filename = f"{frame_path.stem}_tile_{idx:03d}.jpg"
            tile_path = self.output_dir / tile_filename
            cv2.imwrite(str(tile_path), tile)

            tiles_metadata.append(
                {
                    "source_frame": str(frame_path),
                    "tile_path": str(tile_path),
                    "tile_index": idx,
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "tile_size": self.tile_size,
                }
            )

        return tiles_metadata

    def _save_metadata(self, metadata: dict) -> None:
        """Save tile generation metadata."""
        metadata_path = self.output_dir / "metadata.json"

        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
