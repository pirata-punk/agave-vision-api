#!/usr/bin/env python3
"""
Training CLI

Command-line interface for model training and evaluation.
"""

import click
from pathlib import Path

from agave_vision.training.trainer import YOLOTrainer
from agave_vision.training.evaluator import ModelEvaluator, compare_models
from agave_vision.utils.logging import setup_logging


@click.group()
def cli():
    """Agave Vision Training CLI"""
    setup_logging("training-cli", level="INFO", format="text")


@cli.command()
@click.option(
    "--data-yaml",
    type=click.Path(exists=True),
    default="configs/yolo_data.yaml",
    help="Path to data.yaml configuration",
)
@click.option(
    "--model", type=str, default="yolov8n", help="Base model (yolov8n, yolov8s, yolov8m, etc.)"
)
@click.option("--epochs", type=int, default=100, help="Number of training epochs")
@click.option("--imgsz", type=int, default=640, help="Input image size")
@click.option("--batch", type=int, default=16, help="Batch size")
@click.option("--device", type=str, default="cuda", help="Training device (cuda/cpu/mps)")
@click.option("--version", type=str, default=None, help="Version name for this run")
@click.option("--patience", type=int, default=50, help="Early stopping patience")
def train(data_yaml, model, epochs, imgsz, batch, device, version, patience):
    """Train a YOLO model."""
    click.echo(f"Training {model} model...")

    trainer = YOLOTrainer(
        data_yaml=data_yaml, model_name=model, output_dir="training/runs", version_name=version
    )

    results = trainer.train(
        epochs=epochs, imgsz=imgsz, batch=batch, device=device, patience=patience
    )

    click.echo(f"✓ Training complete!")
    click.echo(f"  Version: {trainer.version_name}")
    click.echo(f"  Best weights: {trainer.run_dir}/weights/best.pt")


@cli.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to checkpoint (last.pt)",
)
def resume(checkpoint):
    """Resume training from checkpoint."""
    click.echo(f"Resuming training from {checkpoint}...")

    # Extract version name from checkpoint path
    checkpoint_path = Path(checkpoint)
    version_name = checkpoint_path.parent.parent.name

    trainer = YOLOTrainer(
        data_yaml="configs/yolo_data.yaml",
        output_dir="training/runs",
        version_name=version_name,
    )

    results = trainer.resume(checkpoint_path=checkpoint_path)
    click.echo("✓ Resumed training complete!")


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weights (best.pt)",
)
@click.option(
    "--data-yaml",
    type=click.Path(exists=True),
    default="configs/yolo_data.yaml",
    help="Path to data.yaml",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="test",
    help="Dataset split to evaluate",
)
@click.option("--imgsz", type=int, default=640, help="Input image size")
@click.option("--batch", type=int, default=16, help="Batch size")
def evaluate(model, data_yaml, split, imgsz, batch):
    """Evaluate a trained model."""
    click.echo(f"Evaluating model on {split} set...")

    evaluator = ModelEvaluator(model_path=model, data_yaml=data_yaml)
    metrics = evaluator.evaluate(split=split, imgsz=imgsz, batch=batch)

    click.echo("\n✓ Evaluation Results:")
    click.echo(f"  mAP@0.5:      {metrics['metrics']['map50']:.4f}")
    click.echo(f"  mAP@0.5-0.95: {metrics['metrics']['map50_95']:.4f}")
    click.echo(f"  Precision:    {metrics['metrics']['precision']:.4f}")
    click.echo(f"  Recall:       {metrics['metrics']['recall']:.4f}")


@cli.command()
@click.argument("model_paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--data-yaml",
    type=click.Path(exists=True),
    default="configs/yolo_data.yaml",
    help="Path to data.yaml",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="test",
    help="Dataset split to evaluate",
)
def compare(model_paths, data_yaml, split):
    """Compare multiple models."""
    if len(model_paths) < 2:
        click.echo("Error: Provide at least 2 model paths to compare")
        return

    click.echo(f"Comparing {len(model_paths)} models on {split} set...\n")

    results = compare_models(
        model_paths=[Path(p) for p in model_paths], data_yaml=Path(data_yaml), split=split
    )

    click.echo("\n✓ Comparison complete!")


if __name__ == "__main__":
    cli()
