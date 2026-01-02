"""
Detection Logging

Logs all detection results for analysis and debugging.
Provides rotating storage with configurable retention.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionLogger:
    """
    Log all detections to rotating JSON files or SQLite.

    Useful for debugging, analysis, and model performance tracking.

    Example:
        >>> logger = DetectionLogger(path="data/logs")
        >>> logger.log_detection({
        ...     "timestamp": "2025-01-02T14:30:00",
        ...     "camera_id": "cam1",
        ...     "detections": [...],
        ...     "num_alerts": 2
        ... })
        >>> logs = logger.get_logs(camera_id="cam1", limit=100)
    """

    def __init__(
        self,
        path: str = "data/detection_logs",
        retention_days: int = 7,
        max_entries_per_file: int = 10000,
    ):
        """
        Initialize detection logger.

        Args:
            path: Directory path for log files
            retention_days: Days to retain logs before auto-cleanup
            max_entries_per_file: Maximum entries per rotating log file
        """
        self.log_dir = Path(path)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self.max_entries_per_file = max_entries_per_file

        self.current_file: Optional[Path] = None
        self.current_entries = 0

        # Initialize current log file
        self._init_current_file()

    def _init_current_file(self):
        """Initialize or rotate to new log file."""
        # Find latest log file or create new one
        log_files = sorted(self.log_dir.glob("detections_*.jsonl"), reverse=True)

        if log_files:
            # Check if latest file is full
            latest_file = log_files[0]
            entry_count = sum(1 for _ in latest_file.open())

            if entry_count < self.max_entries_per_file:
                self.current_file = latest_file
                self.current_entries = entry_count
                return

        # Create new log file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.log_dir / f"detections_{timestamp}.jsonl"
        self.current_entries = 0

    def log_detection(self, detection_result: Dict[str, Any]):
        """
        Log a detection result.

        Args:
            detection_result: Detection result dictionary from ML API
                {
                    "timestamp": "...",
                    "camera_id": "...",
                    "detections": [...],
                    "num_alerts": 0,
                    ...
                }
        """
        # Rotate file if needed
        if self.current_entries >= self.max_entries_per_file:
            self._init_current_file()

        # Append to current log file (JSONL format)
        with self.current_file.open("a") as f:
            json.dump(detection_result, f)
            f.write("\n")

        self.current_entries += 1

        # Periodic cleanup of old logs
        if self.current_entries % 1000 == 0:
            self.cleanup_old_logs()

    def get_logs(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve detection logs with filters.

        Args:
            camera_id: Filter by camera ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of log entries

        Returns:
            List of detection result dictionaries
        """
        # Get all log files in reverse chronological order
        log_files = sorted(self.log_dir.glob("detections_*.jsonl"), reverse=True)

        logs = []

        for log_file in log_files:
            # Check if we've hit limit
            if len(logs) >= limit:
                break

            # Read log file (JSONL format)
            with log_file.open("r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Apply filters
                        if camera_id and entry.get("camera_id") != camera_id:
                            continue

                        # Time filters
                        entry_time = datetime.fromisoformat(entry.get("timestamp", ""))

                        if start_time and entry_time < start_time:
                            continue

                        if end_time and entry_time > end_time:
                            continue

                        logs.append(entry)

                        # Check limit
                        if len(logs) >= limit:
                            break

                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        return logs

    def export_logs(
        self,
        output_path: str,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ):
        """
        Export logs to file.

        Args:
            output_path: Output file path
            camera_id: Filter by camera ID
            start_time: Filter by start time
            end_time: Filter by end time
            format: Export format ("json" or "csv")
        """
        logs = self.get_logs(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000000  # Get all matching logs
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with output_file.open("w") as f:
                json.dump({"logs": logs, "count": len(logs)}, f, indent=2)

        elif format == "csv":
            import csv

            # Flatten detection results for CSV
            rows = []
            for log in logs:
                for detection in log.get("detections", []):
                    rows.append({
                        "timestamp": log.get("timestamp"),
                        "camera_id": log.get("camera_id"),
                        "class_name": detection.get("class_name"),
                        "confidence": detection.get("confidence"),
                        "bbox": json.dumps(detection.get("bbox")),
                        "center": json.dumps(detection.get("center")),
                        "is_unknown": detection.get("is_unknown"),
                    })

            if rows:
                with output_file.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def cleanup_old_logs(self) -> int:
        """
        Remove log files older than retention period.

        Returns:
            Number of files deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        deleted_count = 0

        for log_file in self.log_dir.glob("detections_*.jsonl"):
            # Parse timestamp from filename: detections_YYYYMMDD_HHMMSS.jsonl
            try:
                timestamp_str = log_file.stem.split("_", 1)[1]  # Get YYYYMMDD_HHMMSS part
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if file_time < cutoff:
                    log_file.unlink()
                    deleted_count += 1

            except (ValueError, IndexError):
                # Skip files with unexpected names
                continue

        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics.

        Returns:
            {
                "total_log_files": 5,
                "total_entries": 45000,
                "oldest_log": "2025-01-01T10:00:00",
                "newest_log": "2025-01-02T14:30:00",
                "disk_usage_mb": 12.5
            }
        """
        log_files = list(self.log_dir.glob("detections_*.jsonl"))

        if not log_files:
            return {
                "total_log_files": 0,
                "total_entries": 0,
                "oldest_log": None,
                "newest_log": None,
                "disk_usage_mb": 0.0,
            }

        # Count entries
        total_entries = sum(
            sum(1 for _ in log_file.open())
            for log_file in log_files
        )

        # Get timestamps
        timestamps = []
        for log_file in log_files:
            try:
                timestamp_str = log_file.stem.split("_", 1)[1]
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                timestamps.append(file_time)
            except (ValueError, IndexError):
                continue

        oldest = min(timestamps).isoformat() if timestamps else None
        newest = max(timestamps).isoformat() if timestamps else None

        # Calculate disk usage
        disk_usage_bytes = sum(log_file.stat().st_size for log_file in log_files)
        disk_usage_mb = disk_usage_bytes / (1024 * 1024)

        return {
            "total_log_files": len(log_files),
            "total_entries": total_entries,
            "oldest_log": oldest,
            "newest_log": newest,
            "disk_usage_mb": round(disk_usage_mb, 2),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DetectionLogger("
            f"files={stats['total_log_files']}, "
            f"entries={stats['total_entries']}, "
            f"retention={self.retention_days}d)"
        )
