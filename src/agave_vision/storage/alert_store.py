"""
Alert Storage

Simple alert persistence using SQLite or JSON files.
Provides queryable alert history for external systems.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class AlertStore:
    """
    Simple alert storage using SQLite or JSON.

    Stores alert events with queryable filters for external systems
    to retrieve alert history.

    Example:
        >>> store = AlertStore(storage_type="sqlite", path="data/alerts.db")
        >>> alert_id = store.save_alert({
        ...     "camera_id": "cam1",
        ...     "timestamp": "2025-01-02T14:30:00",
        ...     "detection": {...},
        ...     "roi_name": "loading_zone"
        ... })
        >>> alerts = store.get_alerts(camera_id="cam1", limit=10)
    """

    def __init__(
        self,
        storage_type: str = "sqlite",
        path: str = "data/alerts.db",
    ):
        """
        Initialize alert storage.

        Args:
            storage_type: Storage backend ("sqlite" or "json")
            path: Path to storage file
        """
        self.storage_type = storage_type
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if storage_type == "sqlite":
            self._init_sqlite()
        elif storage_type == "json":
            self._init_json()
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                roi_name TEXT,
                violation_type TEXT,
                detected_class TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                center_x REAL,
                center_y REAL,
                is_unknown INTEGER,
                allowed_classes TEXT,
                strict_mode INTEGER,
                alert_data TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_camera_timestamp
            ON alerts(camera_id, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON alerts(timestamp DESC)
        """)

        self.conn.commit()

    def _init_json(self):
        """Initialize JSON file storage."""
        if not self.path.exists():
            self.path.write_text(json.dumps({"alerts": []}, indent=2))

    def save_alert(self, alert: Dict[str, Any]) -> str:
        """
        Save alert and return alert_id.

        Args:
            alert: Alert dictionary from AlertEvent.to_dict()

        Returns:
            Generated alert_id (UUID)
        """
        alert_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        if self.storage_type == "sqlite":
            return self._save_alert_sqlite(alert, alert_id, created_at)
        else:
            return self._save_alert_json(alert, alert_id, created_at)

    def _save_alert_sqlite(
        self,
        alert: Dict[str, Any],
        alert_id: str,
        created_at: str
    ) -> str:
        """Save alert to SQLite."""
        detection = alert.get("detection", {})
        bbox = detection.get("bbox", [0, 0, 0, 0])
        center = detection.get("center", [0, 0])

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (
                alert_id, camera_id, timestamp, roi_name, violation_type,
                detected_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                center_x, center_y, is_unknown, allowed_classes, strict_mode,
                alert_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_id,
            alert.get("camera_id"),
            alert.get("timestamp"),
            alert.get("roi_name"),
            alert.get("violation_type", "forbidden_class"),
            detection.get("class_name"),
            detection.get("confidence"),
            bbox[0], bbox[1], bbox[2], bbox[3],
            center[0], center[1],
            1 if detection.get("is_unknown", False) else 0,
            json.dumps(alert.get("allowed_classes", [])),
            1 if alert.get("strict_mode", True) else 0,
            json.dumps(alert),
            created_at
        ))
        self.conn.commit()

        return alert_id

    def _save_alert_json(
        self,
        alert: Dict[str, Any],
        alert_id: str,
        created_at: str
    ) -> str:
        """Save alert to JSON file."""
        data = json.loads(self.path.read_text())

        alert_record = {
            "alert_id": alert_id,
            "created_at": created_at,
            **alert
        }

        data["alerts"].append(alert_record)

        self.path.write_text(json.dumps(data, indent=2))

        return alert_id

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query alerts with filters.

        Args:
            camera_id: Filter by camera ID
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (inclusive)
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        if self.storage_type == "sqlite":
            return self._get_alerts_sqlite(camera_id, start_time, end_time, limit)
        else:
            return self._get_alerts_json(camera_id, start_time, end_time, limit)

    def _get_alerts_sqlite(
        self,
        camera_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Query alerts from SQLite."""
        query = "SELECT alert_data FROM alerts WHERE 1=1"
        params = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        alerts = []
        for row in cursor.fetchall():
            alert_data = json.loads(row[0])
            alerts.append(alert_data)

        return alerts

    def _get_alerts_json(
        self,
        camera_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Query alerts from JSON file."""
        data = json.loads(self.path.read_text())
        alerts = data.get("alerts", [])

        # Filter alerts
        filtered = []
        for alert in alerts:
            # Camera filter
            if camera_id and alert.get("camera_id") != camera_id:
                continue

            # Time filters
            alert_time = datetime.fromisoformat(alert.get("timestamp", ""))

            if start_time and alert_time < start_time:
                continue

            if end_time and alert_time > end_time:
                continue

            filtered.append(alert)

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit results
        return filtered[:limit]

    def clear_old_alerts(self, days: int = 30) -> int:
        """
        Remove alerts older than N days.

        Args:
            days: Number of days to retain

        Returns:
            Number of alerts deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        if self.storage_type == "sqlite":
            return self._clear_old_alerts_sqlite(cutoff)
        else:
            return self._clear_old_alerts_json(cutoff)

    def _clear_old_alerts_sqlite(self, cutoff: datetime) -> int:
        """Clear old alerts from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM alerts WHERE timestamp < ?",
            (cutoff.isoformat(),)
        )
        self.conn.commit()
        return cursor.rowcount

    def _clear_old_alerts_json(self, cutoff: datetime) -> int:
        """Clear old alerts from JSON file."""
        data = json.loads(self.path.read_text())
        alerts = data.get("alerts", [])

        before_count = len(alerts)

        # Keep only recent alerts
        recent_alerts = [
            alert for alert in alerts
            if datetime.fromisoformat(alert.get("timestamp", "")) >= cutoff
        ]

        data["alerts"] = recent_alerts
        self.path.write_text(json.dumps(data, indent=2))

        return before_count - len(recent_alerts)

    def get_alert_count(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Get count of alerts matching filters.

        Args:
            camera_id: Filter by camera ID
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Number of matching alerts
        """
        if self.storage_type == "sqlite":
            query = "SELECT COUNT(*) FROM alerts WHERE 1=1"
            params = []

            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()[0]
        else:
            # For JSON, just get all and count
            alerts = self.get_alerts(camera_id, start_time, end_time, limit=1000000)
            return len(alerts)

    def close(self):
        """Close storage connection (SQLite only)."""
        if self.storage_type == "sqlite" and hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __repr__(self) -> str:
        return f"AlertStore(type={self.storage_type}, path={self.path})"
