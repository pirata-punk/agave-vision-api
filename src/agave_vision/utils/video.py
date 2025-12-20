"""
Video I/O Utilities

Provides OpenCV wrappers and video stream helpers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoCapture:
    """
    Wrapper around cv2.VideoCapture with reconnection logic.

    Args:
        source: Video source (file path, RTSP URL, or camera index)
        timeout: Read timeout in seconds
        reconnect_delay: Delay between reconnection attempts

    Example:
        >>> cap = VideoCapture("rtsp://camera/stream")
        >>> for frame in cap:
        ...     process(frame)
    """

    def __init__(
        self,
        source: str | int,
        timeout: float = 30.0,
        reconnect_delay: float = 5.0,
    ):
        self.source = source
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time: float = 0

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if opened successfully
        """
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._last_frame_time = time.time()

        return ret, frame

    def get_fps(self) -> float:
        """Get source FPS (0 if unknown)."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    def get_size(self) -> Tuple[int, int]:
        """
        Get frame size (width, height).

        Returns:
            Tuple of (width, height), or (0, 0) if unknown
        """
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        """Check if video source is opened."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release the video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __iter__(self):
        """Iterator interface."""
        return self

    def __next__(self) -> np.ndarray:
        """
        Get next frame (iterator interface).

        Raises:
            StopIteration: If no more frames available
        """
        ret, frame = self.read()
        if not ret or frame is None:
            raise StopIteration
        return frame


class VideoWriter:
    """
    Wrapper around cv2.VideoWriter.

    Args:
        output_path: Output file path
        fps: Frames per second
        frame_size: (width, height)
        fourcc: FourCC codec (default: mp4v)

    Example:
        >>> writer = VideoWriter("output.mp4", fps=30, frame_size=(640, 480))
        >>> writer.write(frame)
        >>> writer.release()
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        frame_size: Tuple[int, int] = (640, 480),
        fourcc: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer: Optional[cv2.VideoWriter] = None

    def open(self) -> bool:
        """
        Open the video writer.

        Returns:
            True if opened successfully
        """
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            self.fourcc,
            self.fps,
            self.frame_size,
        )
        return self.writer.isOpened()

    def write(self, frame: np.ndarray):
        """
        Write a frame to the video.

        Args:
            frame: Frame to write
        """
        if self.writer is None:
            self.open()

        if self.writer is not None and self.writer.isOpened():
            # Resize frame if needed
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                frame = cv2.resize(frame, self.frame_size)

            self.writer.write(frame)

    def release(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def get_video_info(source: str | int) -> dict:
    """
    Get video metadata.

    Args:
        source: Video source (file or URL)

    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info
