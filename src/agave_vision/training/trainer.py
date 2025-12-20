"""
YOLO Trainer

Orchestrates YOLOv8 model training with configuration management.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class YOLOTrainer:
    """
    YOLO model trainer with configuration and versioning.

    Args:
        data_yaml: Path to YOLO data.yaml configuration
        model_name: Base model name (e.g., "yolov8n", "yolov8s")
        output_dir: Directory for training outputs
        version_name: Version name for this training run

    Example:
        >>> trainer = YOLOTrainer(
        ...     data_yaml="configs/yolo_data.yaml",
        ...     model_name="yolov8n",
        ...     output_dir="training/runs",
        ...     version_name="v1_baseline"
        ... )
        >>> results = trainer.train(epochs=100)
    """

    def __init__(
        self,
        data_yaml: str | Path,
        model_name: str = "yolov8n",
        output_dir: str | Path = "training/runs",
        version_name: Optional[str] = None,
    ):
        self.data_yaml = Path(data_yaml)
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # Generate version name if not provided
        if version_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"{model_name}_{timestamp}"

        self.version_name = version_name
        self.run_dir = self.output_dir / version_name

        # Ensure output directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        workers: int = 8,
        device: str = "cuda",
        patience: int = 50,
        save_period: int = 10,
        **kwargs,
    ) -> dict:
        """
        Train YOLO model.

        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            workers: Number of data loading workers
            device: Training device (cuda/cpu/mps)
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional arguments for YOLO.train()

        Returns:
            Training results dictionary
        """
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml}")

        logger.info(f"Starting training: {self.version_name}")
        logger.info(f"Data: {self.data_yaml}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output: {self.run_dir}")

        # Initialize model
        model = YOLO(f"{self.model_name}.pt")

        # Save training configuration
        training_config = {
            "version_name": self.version_name,
            "model_name": self.model_name,
            "data_yaml": str(self.data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": device,
            "patience": patience,
            "save_period": save_period,
            "started_at": datetime.now().isoformat(),
        }
        self._save_config(training_config)

        # Train
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            patience=patience,
            save_period=save_period,
            project=str(self.output_dir),
            name=self.version_name,
            exist_ok=True,
            **kwargs,
        )

        logger.info(f"Training complete: {self.version_name}")
        logger.info(f"Best model: {self.run_dir}/weights/best.pt")

        # Save final results
        training_config["completed_at"] = datetime.now().isoformat()
        training_config["best_weights"] = str(self.run_dir / "weights" / "best.pt")
        self._save_config(training_config)

        return results

    def resume(self, checkpoint_path: Optional[str | Path] = None) -> dict:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (default: last.pt in run_dir)

        Returns:
            Training results dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.run_dir / "weights" / "last.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming training from: {checkpoint_path}")

        model = YOLO(str(checkpoint_path))
        results = model.train(resume=True)

        logger.info(f"Resumed training complete: {self.version_name}")

        return results

    def _save_config(self, config: dict) -> None:
        """Save training configuration."""
        config_path = self.run_dir / "training_config.json"

        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        logger.debug(f"Saved training config to {config_path}")
