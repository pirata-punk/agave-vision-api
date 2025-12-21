#!/usr/bin/env python3
"""
Ingestion CLI

Command-line interface for data ingestion pipelines.
"""

import click
from pathlib import Path

from agave_vision.ingestion.static.video_processor import VideoProcessor
from agave_vision.ingestion.static.tile_generator import TileGenerator
from agave_vision.ingestion.static.dataset_builder import DatasetBuilder
from agave_vision.utils.logging import setup_logging


@click.group()
def cli():
    """Agave Vision Data Ingestion CLI"""
    setup_logging("ingestion-cli", level="INFO", format="text")


@cli.command()
@click.option(
    "--video-dir",
    type=click.Path(exists=True),
    default="data/videos",
    help="Directory containing video files",
)
@click.option(
    "--output-dir", type=click.Path(), default="data/frames", help="Output directory for frames"
)
@click.option("--sample-rate", type=int, default=30, help="Extract every Nth frame")
@click.option(
    "--dedup-threshold", type=float, default=0.95, help="Similarity threshold for deduplication"
)
@click.option(
    "--min-sharpness", type=float, default=100.0, help="Minimum sharpness score to keep frame"
)
def extract_frames(video_dir, output_dir, sample_rate, dedup_threshold, min_sharpness):
    """Extract frames from videos with deduplication."""
    click.echo(f"Extracting frames from {video_dir}...")

    processor = VideoProcessor(
        video_dir=video_dir,
        output_dir=output_dir,
        sample_rate=sample_rate,
        dedup_threshold=dedup_threshold,
        min_sharpness=min_sharpness,
    )

    metadata = processor.process_all_videos()
    click.echo(f"✓ Extracted {len(metadata)} frames to {output_dir}")


@cli.command()
@click.option(
    "--frames-dir",
    type=click.Path(exists=True),
    default="data/frames",
    help="Directory containing frames",
)
@click.option(
    "--output-dir", type=click.Path(), default="data/tiles_pool", help="Output directory for tiles"
)
@click.option("--tile-size", type=int, default=640, help="Size of each tile (square)")
@click.option("--overlap", type=int, default=128, help="Overlap between tiles")
def generate_tiles(frames_dir, output_dir, tile_size, overlap):
    """Generate tiles from frames using sliding window."""
    click.echo(f"Generating tiles from {frames_dir}...")

    generator = TileGenerator(
        frames_dir=frames_dir, output_dir=output_dir, tile_size=tile_size, overlap=overlap
    )

    metadata = generator.generate_all_tiles()
    click.echo(f"✓ Generated {metadata['total_tiles']} tiles to {output_dir}")


@cli.command()
@click.option(
    "--rounds-dir",
    type=click.Path(exists=True),
    default="data/tiles_pool/tiles_man",
    help="Directory containing tiles_round{1..4}",
)
@click.option(
    "--output-dir", type=click.Path(), default="data/tiles_yolo", help="Output directory for dataset"
)
@click.option(
    "--classes",
    multiple=True,
    default=["object", "pine", "worker"],
    help="Class names in order (repeatable)",
)
@click.option("--train-ratio", type=float, default=0.7, help="Training set ratio")
@click.option("--val-ratio", type=float, default=0.15, help="Validation set ratio")
@click.option("--test-ratio", type=float, default=0.15, help="Test set ratio")
@click.option("--seed", type=int, default=123, help="Random seed for reproducibility")
def build_dataset(rounds_dir, output_dir, classes, train_ratio, val_ratio, test_ratio, seed):
    """Build YOLO dataset from labeled tiles with train/val/test splits."""
    click.echo(f"Building dataset from {rounds_dir}...")

    builder = DatasetBuilder(
        rounds_dir=rounds_dir,
        output_dir=output_dir,
        class_names=list(classes),
        split_ratios=(train_ratio, val_ratio, test_ratio),
        random_seed=seed,
    )

    metadata = builder.build_dataset()
    click.echo(f"✓ Built dataset with {metadata['total']} samples:")
    click.echo(f"  - Train: {metadata['splits']['train']}")
    click.echo(f"  - Val: {metadata['splits']['val']}")
    click.echo(f"  - Test: {metadata['splits']['test']}")


@cli.command()
def pipeline():
    """Run complete ingestion pipeline (videos → frames → tiles → dataset)."""
    click.echo("Running complete ingestion pipeline...")

    # Step 1: Extract frames
    click.echo("\n[1/3] Extracting frames...")
    processor = VideoProcessor(video_dir="data/videos", output_dir="data/frames")
    frames_metadata = processor.process_all_videos()
    click.echo(f"✓ Extracted {len(frames_metadata)} frames")

    # Step 2: Generate tiles
    click.echo("\n[2/3] Generating tiles...")
    generator = TileGenerator(frames_dir="data/frames", output_dir="data/tiles_pool")
    tiles_metadata = generator.generate_all_tiles()
    click.echo(f"✓ Generated {tiles_metadata['total_tiles']} tiles")

    # Step 3: Build dataset (requires manual labeling in between)
    click.echo("\n[3/3] Dataset building...")
    click.echo("⚠ Manual labeling required before building dataset")
    click.echo("  1. Label tiles in data/tiles_pool")
    click.echo("  2. Organize into tiles_round{1..4} directories")
    click.echo("  3. Run: ingestion build-dataset")


if __name__ == "__main__":
    cli()
