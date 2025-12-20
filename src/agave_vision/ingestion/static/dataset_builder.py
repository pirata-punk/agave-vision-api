"""
Dataset Builder

Builds YOLO-format datasets from labeled tiles with train/val/test splits.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm

from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageLabelPair:
    """Represents an image/label pair for the dataset."""

    image_path: Path
    label_path: Path
    source_round: str
    new_name: str


class DatasetBuilder:
    """
    Build YOLO dataset from labeled tiles with train/val/test splits.

    Args:
        rounds_dir: Directory containing tiles_round{1..4} subdirectories
        output_dir: Output directory for YOLO dataset
        class_names: List of class names in order (matches YOLO IDs)
        split_ratios: Tuple of (train, val, test) ratios
        random_seed: Random seed for reproducible splits
        allow_empty_labels: Whether to include images with empty label files

    Example:
        >>> builder = DatasetBuilder(
        ...     rounds_dir="data/tiles_pool/tiles_man",
        ...     output_dir="data/tiles_yolo",
        ...     class_names=["object", "pine", "worker"]
        ... )
        >>> builder.build_dataset()
    """

    def __init__(
        self,
        rounds_dir: str | Path,
        output_dir: str | Path,
        class_names: List[str],
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 123,
        allow_empty_labels: bool = True,
    ):
        self.rounds_dir = Path(rounds_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        self.allow_empty_labels = allow_empty_labels

    def build_dataset(self) -> dict:
        """
        Build complete YOLO dataset.

        Returns:
            Metadata dictionary with split statistics
        """
        # Gather all image/label pairs
        pairs = self._gather_pairs()

        if not pairs:
            logger.error("No image/label pairs found")
            return {}

        logger.info(f"Found {len(pairs)} image/label pairs")

        # Split into train/val/test
        splits = self._split_pairs(pairs)

        logger.info(
            f"Split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        # Copy files to output structure
        for split_name, split_pairs in splits.items():
            self._copy_split(split_name, split_pairs)

        # Generate data.yaml
        self._generate_data_yaml()

        # Save metadata
        metadata = self._generate_metadata(splits)
        self._save_metadata(metadata)

        logger.info(f"Dataset built successfully at {self.output_dir}")

        return metadata

    def _gather_pairs(self) -> List[ImageLabelPair]:
        """Gather all valid image/label pairs from round directories."""
        pairs = []
        round_names = [
            "tiles_round1",
            "tiles_round2",
            "tiles_round3",
            "tiles_round4",
        ]

        for round_name in round_names:
            round_dir = self.rounds_dir / round_name
            if not round_dir.exists():
                logger.warning(f"Round directory not found: {round_dir}")
                continue

            images_dir = round_dir / "images"
            labels_dir = round_dir / "labels"

            if not images_dir.exists() or not labels_dir.exists():
                logger.warning(f"Missing images or labels in {round_name}")
                continue

            round_pairs = self._gather_round_pairs(round_name, images_dir, labels_dir)
            pairs.extend(round_pairs)
            logger.info(f"Found {len(round_pairs)} pairs in {round_name}")

        return pairs

    def _gather_round_pairs(
        self, round_name: str, images_dir: Path, labels_dir: Path
    ) -> List[ImageLabelPair]:
        """Gather pairs from a single round directory."""
        pairs = []
        seen_names = {}

        for img_path in images_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Find corresponding label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Check if label is empty
            if not self.allow_empty_labels and label_path.stat().st_size == 0:
                continue

            # Generate unique name
            base_name = f"{round_name}_{img_path.stem}"
            if base_name in seen_names:
                seen_names[base_name] += 1
                new_name = f"{base_name}_{seen_names[base_name]}"
            else:
                seen_names[base_name] = 0
                new_name = base_name

            pairs.append(
                ImageLabelPair(
                    image_path=img_path,
                    label_path=label_path,
                    source_round=round_name,
                    new_name=new_name,
                )
            )

        return pairs

    def _split_pairs(self, pairs: List[ImageLabelPair]) -> dict:
        """Split pairs into train/val/test."""
        random.seed(self.random_seed)
        random.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        return {
            "train": pairs[:n_train],
            "val": pairs[n_train : n_train + n_val],
            "test": pairs[n_train + n_val :],
        }

    def _copy_split(self, split_name: str, pairs: List[ImageLabelPair]) -> None:
        """Copy image/label pairs to split directory."""
        images_dir = self.output_dir / "images" / split_name
        labels_dir = self.output_dir / "labels" / split_name

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for pair in tqdm(pairs, desc=f"Copying {split_name}"):
            # Copy image
            new_image_path = images_dir / f"{pair.new_name}.jpg"
            shutil.copy2(pair.image_path, new_image_path)

            # Copy label
            new_label_path = labels_dir / f"{pair.new_name}.txt"
            shutil.copy2(pair.label_path, new_label_path)

    def _generate_data_yaml(self) -> None:
        """Generate YOLO data.yaml configuration."""
        data_yaml_path = self.output_dir.parent / "configs" / "yolo_data.yaml"
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        with data_yaml_path.open("w") as f:
            yaml.dump(data, f, sort_keys=False)

        logger.info(f"Generated data.yaml at {data_yaml_path}")

    def _generate_metadata(self, splits: dict) -> dict:
        """Generate dataset metadata."""
        return {
            "rounds_dir": str(self.rounds_dir),
            "output_dir": str(self.output_dir),
            "class_names": self.class_names,
            "split_ratios": self.split_ratios,
            "random_seed": self.random_seed,
            "splits": {name: len(pairs) for name, pairs in splits.items()},
            "total": sum(len(pairs) for pairs in splits.values()),
        }

    def _save_metadata(self, metadata: dict) -> None:
        """Save dataset metadata."""
        metadata_path = self.output_dir / "metadata.json"

        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
