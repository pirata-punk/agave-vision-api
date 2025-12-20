"""
Model Evaluator

Evaluates trained YOLO models on test sets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate YOLO models on test/validation sets.

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to YOLO data.yaml configuration

    Example:
        >>> evaluator = ModelEvaluator(
        ...     model_path="models/v1_baseline/weights/best.pt",
        ...     data_yaml="configs/yolo_data.yaml"
        ... )
        >>> results = evaluator.evaluate(split="test")
    """

    def __init__(self, model_path: str | Path, data_yaml: str | Path):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml}")

        # Load model
        self.model = YOLO(str(self.model_path))

    def evaluate(
        self, split: str = "test", imgsz: int = 640, batch: int = 16, save_json: bool = True
    ) -> dict:
        """
        Evaluate model on specified split.

        Args:
            split: Dataset split to evaluate ("train", "val", or "test")
            imgsz: Input image size
            batch: Batch size
            save_json: Whether to save results as JSON

        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating model on {split} set...")
        logger.info(f"Model: {self.model_path}")

        # Run validation
        results = self.model.val(data=str(self.data_yaml), split=split, imgsz=imgsz, batch=batch)

        # Extract metrics
        metrics = {
            "model_path": str(self.model_path),
            "data_yaml": str(self.data_yaml),
            "split": split,
            "imgsz": imgsz,
            "metrics": {
                "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
                "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
            },
        }

        # Per-class metrics
        if hasattr(results, "box") and hasattr(results.box, "class_result"):
            metrics["per_class"] = results.box.class_result

        logger.info(f"Evaluation complete:")
        logger.info(f"  mAP@0.5: {metrics['metrics']['map50']:.4f}")
        logger.info(f"  mAP@0.5-0.95: {metrics['metrics']['map50_95']:.4f}")
        logger.info(f"  Precision: {metrics['metrics']['precision']:.4f}")
        logger.info(f"  Recall: {metrics['metrics']['recall']:.4f}")

        if save_json:
            self._save_metrics(metrics)

        return metrics

    def _save_metrics(self, metrics: dict) -> None:
        """Save evaluation metrics to JSON."""
        metrics_dir = self.model_path.parent.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        split = metrics["split"]
        metrics_path = metrics_dir / f"metrics_{split}.json"

        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")


def compare_models(model_paths: list[Path], data_yaml: Path, split: str = "test") -> dict:
    """
    Compare multiple models on the same dataset.

    Args:
        model_paths: List of model weight paths
        data_yaml: Path to YOLO data.yaml
        split: Dataset split to evaluate

    Returns:
        Comparison results dictionary
    """
    logger.info(f"Comparing {len(model_paths)} models on {split} set...")

    results = {}

    for model_path in model_paths:
        evaluator = ModelEvaluator(model_path=model_path, data_yaml=data_yaml)
        metrics = evaluator.evaluate(split=split, save_json=False)
        results[model_path.stem] = metrics["metrics"]

    # Print comparison table
    logger.info("\nModel Comparison:")
    logger.info(f"{'Model':<30} {'mAP@0.5':<10} {'mAP@0.5-0.95':<15} {'Precision':<12} {'Recall':<10}")
    logger.info("-" * 85)

    for model_name, metrics in results.items():
        logger.info(
            f"{model_name:<30} "
            f"{metrics['map50']:<10.4f} "
            f"{metrics['map50_95']:<15.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<10.4f}"
        )

    return results
