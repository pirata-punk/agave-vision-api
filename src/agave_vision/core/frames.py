"""
Frame Processing Utilities

Provides utilities for frame preprocessing, tiling, and deduplication.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean


def sliding_window_tiles(
    image: np.ndarray, tile_size: int = 640, overlap: int = 128
) -> Iterator[Tuple[np.ndarray, int, int]]:
    """
    Generate sliding window tiles from an image.

    Args:
        image: Input image (H, W, C)
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles

    Yields:
        Tuple of (tile, x_offset, y_offset)
    """
    height, width = image.shape[:2]
    stride = tile_size - overlap

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Extract tile
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            tile = image[y:y_end, x:x_end]

            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            yield tile, x, y


def compute_frame_sharpness(frame: np.ndarray) -> float:
    """
    Compute frame sharpness using Laplacian variance.

    Higher values indicate sharper images. Useful for frame quality filtering.

    Args:
        frame: Input image (BGR or grayscale)

    Returns:
        Sharpness score (variance of Laplacian)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_frame_brightness(frame: np.ndarray) -> float:
    """
    Compute average brightness of a frame.

    Args:
        frame: Input image (BGR or grayscale)

    Returns:
        Average pixel intensity (0-255)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    return float(gray.mean())


def is_similar_frame(
    frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.95, method: str = "hist"
) -> bool:
    """
    Check if two frames are similar using histogram comparison.

    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
        threshold: Similarity threshold (0.0-1.0), higher means more similar required
        method: Comparison method ("hist" for histogram, "ssim" for structural similarity)

    Returns:
        True if frames are similar above threshold
    """
    if method == "hist":
        # Histogram comparison (faster)
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()

        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation >= threshold

    elif method == "ssim":
        # Structural similarity (more accurate but slower)
        try:
            from skimage.metrics import structural_similarity as ssim

            # Resize to same size if different
            if frame1.shape != frame2.shape:
                h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
                frame1 = cv2.resize(frame1, (w, h))
                frame2 = cv2.resize(frame2, (w, h))

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            score, _ = ssim(gray1, gray2, full=True)
            return score >= threshold
        except ImportError:
            # Fall back to histogram if scikit-image not available
            return is_similar_frame(frame1, frame2, threshold, method="hist")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'hist' or 'ssim'.")


def resize_keep_aspect(
    image: np.ndarray, target_size: int, pad: bool = False
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image keeping aspect ratio.

    Args:
        image: Input image
        target_size: Target size for the longer edge
        pad: If True, pad to square; if False, resize only

    Returns:
        Tuple of (resized_image, (new_width, new_height))
    """
    height, width = image.shape[:2]
    aspect = width / height

    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect)
    else:
        new_height = target_size
        new_width = int(target_size * aspect)

    resized = cv2.resize(image, (new_width, new_height))

    if pad:
        # Pad to square
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
        return padded, (target_size, target_size)

    return resized, (new_width, new_height)


def draw_detection_box(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection bounding box and label on frame.

    Args:
        frame: Input frame (will be modified in-place)
        bbox: Bounding box (x1, y1, x2, y2)
        label: Class label
        confidence: Detection confidence
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Frame with drawn detection
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label
    label_text = f"{label} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

    # Background for text
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1,  # Filled
    )

    # Text
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame
