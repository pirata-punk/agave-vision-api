"""
Video Processor

Extracts and deduplicates frames from video files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from tqdm import tqdm

from agave_vision.core.frames import compute_frame_sharpness, is_similar_frame
from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for an extracted frame."""

    video_name: str
    frame_number: int
    timestamp_ms: float
    output_path: Path
    sharpness: float
    is_duplicate: bool = False


class VideoProcessor:
    """
    Process videos to extract high-quality frames.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame (default: 30 = 1 FPS at 30 FPS video)
        dedup_threshold: Similarity threshold for deduplication (0.0-1.0)
        min_sharpness: Minimum sharpness score to keep frame

    Example:
        >>> processor = VideoProcessor(
        ...     video_dir="data/videos",
        ...     output_dir="data/frames"
        ... )
        >>> metadata = processor.process_all_videos()
    """

    def __init__(
        self,
        video_dir: str | Path,
        output_dir: str | Path,
        sample_rate: int = 30,
        dedup_threshold: float = 0.95,
        min_sharpness: float = 100.0,
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.dedup_threshold = dedup_threshold
        self.min_sharpness = min_sharpness

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_all_videos(self) -> list[FrameMetadata]:
        """
        Process all videos in the video directory.

        Returns:
            List of frame metadata for all extracted frames
        """
        video_files = list(self.video_dir.glob("*.mp4")) + list(self.video_dir.glob("*.avi"))

        if not video_files:
            logger.warning(f"No video files found in {self.video_dir}")
            return []

        logger.info(f"Found {len(video_files)} video files")

        all_metadata = []
        for video_file in video_files:
            metadata = self.process_video(video_file)
            all_metadata.extend(metadata)

        logger.info(f"Extracted {len(all_metadata)} frames total")

        # Save manifest
        self._save_manifest(all_metadata)

        return all_metadata

    def process_video(self, video_path: Path) -> list[FrameMetadata]:
        """
        Process a single video file.

        Args:
            video_path: Path to video file

        Returns:
            List of frame metadata
        """
        logger.info(f"Processing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {video_path.name}, FPS: {fps}, Total frames: {total_frames}")

        metadata = []
        previous_frame = None
        frame_count = 0

        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                # Sample frames
                if frame_count % self.sample_rate != 0:
                    continue

                # Check sharpness
                sharpness = compute_frame_sharpness(frame)
                if sharpness < self.min_sharpness:
                    continue

                # Check for duplicates
                is_dup = False
                if previous_frame is not None:
                    is_dup = is_similar_frame(
                        frame, previous_frame, threshold=self.dedup_threshold, method="hist"
                    )

                if not is_dup:
                    # Save frame
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    output_path = self._save_frame(video_path.stem, frame_count, frame)

                    metadata.append(
                        FrameMetadata(
                            video_name=video_path.name,
                            frame_number=frame_count,
                            timestamp_ms=timestamp_ms,
                            output_path=output_path,
                            sharpness=sharpness,
                            is_duplicate=False,
                        )
                    )

                    previous_frame = frame.copy()

        cap.release()

        logger.info(
            f"Extracted {len(metadata)} frames from {video_path.name} "
            f"(sampled {total_frames // self.sample_rate} total)"
        )

        return metadata

    def _save_frame(self, video_name: str, frame_number: int, frame: np.ndarray) -> Path:
        """Save frame to disk."""
        filename = f"{video_name}_frame_{frame_number:06d}.jpg"
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), frame)
        return output_path

    def _save_manifest(self, metadata: list[FrameMetadata]) -> None:
        """Save frame manifest as JSON."""
        manifest_path = self.output_dir / "frames_manifest.json"

        manifest_data = {
            "total_frames": len(metadata),
            "videos_processed": len(set(m.video_name for m in metadata)),
            "frames": [
                {
                    "video_name": m.video_name,
                    "frame_number": m.frame_number,
                    "timestamp_ms": m.timestamp_ms,
                    "output_path": str(m.output_path),
                    "sharpness": m.sharpness,
                }
                for m in metadata
            ],
        }

        with manifest_path.open("w") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Saved manifest to {manifest_path}")
