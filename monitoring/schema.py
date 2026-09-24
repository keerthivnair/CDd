"""
Data schemas for Task 4: Drift & System Metrics Monitoring.
Directly models the real output of Task 3 (DriftLens) and implements the schema
defined in Section 4.7 of the CDd Proposal.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union, Dict, Any


@dataclass
class WindowDriftResult:
    """
    Represents the output of a single streaming window evaluation.
    Directly consumes real Task 3 output fields and supplements with host performance metrics.
    """
    timestamp: Union[datetime, str, int, float]
    window_id: Union[str, int]
    driftlens_score: float
    driftlens_alarm: bool
    processing_latency_ms: float
    throughput_posts_per_sec: float
    driftlens_threshold: Optional[float] = None
    n_used: Optional[int] = 0
    post_count: Optional[int] = 0
    is_reference: Optional[bool] = False
    mcddd_score: Optional[float] = None
    mcddd_alarm: Optional[bool] = False
    cpu_pct: Optional[float] = None
    mem_mb: Optional[float] = None
    pipeline: str = "cdd-stream"

    @classmethod
    def from_task3_dict(cls, data: Dict[str, Any], default_latency: float = 0.0, default_throughput: float = 0.0) -> "WindowDriftResult":
        """
        Validates and constructs a WindowDriftResult from Task 3's real dictionary output:
        {
            "window_id": 18402,
            "window_start": "2020-05-20T00:00:00",
            "driftlens_score": 0.029159,
            "driftlens_alarm": true,
            "threshold": 0.028602,
            "n_posts": 1445,
            "n_used": 1000,
            "is_reference": false,
            "processing_latency_ms": 234.5,
            "throughput_posts_per_sec": 616.2
        }
        """
        if "window_id" not in data or data["window_id"] is None:
            raise ValueError("Task 3 result missing required field: 'window_id'")
        
        if "driftlens_score" not in data or data["driftlens_score"] is None:
            raise ValueError(f"Task 3 result for window {data.get('window_id')} missing required field: 'driftlens_score'")
        
        try:
            score = float(data["driftlens_score"])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid non-numeric driftlens_score: {data.get('driftlens_score')}")

        if "driftlens_alarm" not in data:
            raise ValueError(f"Task 3 result for window {data.get('window_id')} missing required field: 'driftlens_alarm'")
        alarm = bool(data["driftlens_alarm"])

        ts_val = data.get("window_start") or data.get("timestamp")
        if not ts_val:
            raise ValueError(f"Task 3 result for window {data.get('window_id')} missing timestamp / window_start")

        thr = data.get("threshold", data.get("driftlens_threshold"))
        threshold = float(thr) if thr is not None else None

        post_count = int(data.get("n_posts", data.get("post_count", 0)))
        n_used = int(data.get("n_used", 0))

        latency = float(data.get("processing_latency_ms", default_latency))
        throughput = float(data.get("throughput_posts_per_sec", default_throughput))

        if throughput <= 0.0 and latency > 0.0 and post_count > 0:
            throughput = round(post_count / (latency / 1000.0), 2)

        return cls(
            timestamp=ts_val,
            window_id=data["window_id"],
            driftlens_score=score,
            driftlens_alarm=alarm,
            driftlens_threshold=threshold,
            processing_latency_ms=latency,
            throughput_posts_per_sec=throughput,
            post_count=post_count,
            n_used=n_used,
            is_reference=bool(data.get("is_reference", False)),
            mcddd_score=float(data["mcddd_score"]) if data.get("mcddd_score") is not None else None,
            mcddd_alarm=bool(data["mcddd_alarm"]) if data.get("mcddd_alarm") is not None else False,
            cpu_pct=float(data["cpu_pct"]) if data.get("cpu_pct") is not None else None,
            mem_mb=float(data["mem_mb"]) if data.get("mem_mb") is not None else None,
            pipeline=str(data.get("pipeline", "cdd-stream"))
        )

    def get_datetime(self) -> datetime:
        """Normalizes timestamp to a timezone-aware UTC datetime."""
        if isinstance(self.timestamp, datetime):
            if self.timestamp.tzinfo is None:
                return self.timestamp.replace(tzinfo=timezone.utc)
            return self.timestamp.astimezone(timezone.utc)
        elif isinstance(self.timestamp, (int, float)):
            return datetime.fromtimestamp(float(self.timestamp), tz=timezone.utc)
        elif isinstance(self.timestamp, str):
            try:
                return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except ValueError:
                return datetime.strptime(self.timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    def to_line_protocol(self, measurement: str = "drift_metrics") -> str:
        """Serializes record to InfluxDB Line Protocol string."""
        tags = [
            f"pipeline={self.pipeline}",
            f"window_id={self.window_id}",
            f"driftlens_alarm_state={'drift' if self.driftlens_alarm else 'normal'}",
            f"is_reference={'true' if self.is_reference else 'false'}"
        ]
        tag_str = ",".join(tags)

        fields = [
            f"driftlens_score={float(self.driftlens_score)}",
            f"driftlens_alarm={1 if self.driftlens_alarm else 0}i",
            f"processing_latency_ms={float(self.processing_latency_ms)}",
            f"throughput_posts_per_sec={float(self.throughput_posts_per_sec)}"
        ]
        if self.driftlens_threshold is not None:
            fields.append(f"driftlens_threshold={float(self.driftlens_threshold)}")
        if self.mcddd_score is not None:
            fields.append(f"mcddd_score={float(self.mcddd_score)}")
            fields.append(f"mcddd_alarm={1 if self.mcddd_alarm else 0}i")
        if self.cpu_pct is not None:
            fields.append(f"cpu_pct={float(self.cpu_pct)}")
        if self.mem_mb is not None:
            fields.append(f"mem_mb={float(self.mem_mb)}")
        if self.post_count is not None:
            fields.append(f"post_count={int(self.post_count)}i")
        if self.n_used is not None:
            fields.append(f"n_used={int(self.n_used)}i")

        field_str = ",".join(fields)
        ts_sec = int(self.get_datetime().timestamp())
        return f"{measurement},{tag_str} {field_str} {ts_sec}"

    def to_point(self, measurement: str = "drift_metrics"):
        """Converts record to an InfluxDB Point instance if influxdb_client is present."""
        try:
            from influxdb_client import Point
            dt = self.get_datetime()
            point = (
                Point(measurement)
                .time(dt)
                .tag("pipeline", self.pipeline)
                .tag("window_id", str(self.window_id))
                .tag("driftlens_alarm_state", "drift" if self.driftlens_alarm else "normal")
                .tag("is_reference", "true" if self.is_reference else "false")
                .field("driftlens_score", float(self.driftlens_score))
                .field("driftlens_alarm", 1 if self.driftlens_alarm else 0)
                .field("processing_latency_ms", float(self.processing_latency_ms))
                .field("throughput_posts_per_sec", float(self.throughput_posts_per_sec))
            )
            if self.driftlens_threshold is not None:
                point.field("driftlens_threshold", float(self.driftlens_threshold))
            if self.mcddd_score is not None:
                point.field("mcddd_score", float(self.mcddd_score))
                point.field("mcddd_alarm", 1 if self.mcddd_alarm else 0)
            if self.cpu_pct is not None:
                point.field("cpu_pct", float(self.cpu_pct))
            if self.mem_mb is not None:
                point.field("mem_mb", float(self.mem_mb))
            if self.post_count is not None:
                point.field("post_count", int(self.post_count))
            if self.n_used is not None:
                point.field("n_used", int(self.n_used))
            return point
        except ImportError:
            return None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Converts record to a dictionary suitable for CSV serialization."""
        return {
            "timestamp": self.get_datetime().isoformat(),
            "window_id": str(self.window_id),
            "driftlens_score": round(float(self.driftlens_score), 6),
            "driftlens_alarm": int(self.driftlens_alarm),
            "driftlens_threshold": round(float(self.driftlens_threshold), 6) if self.driftlens_threshold is not None else "",
            "mcddd_score": round(float(self.mcddd_score), 6) if self.mcddd_score is not None else "",
            "mcddd_alarm": int(self.mcddd_alarm) if self.mcddd_alarm is not None else "",
            "processing_latency_ms": round(float(self.processing_latency_ms), 2),
            "throughput_posts_per_sec": round(float(self.throughput_posts_per_sec), 2),
            "cpu_pct": round(float(self.cpu_pct), 2) if self.cpu_pct is not None else "",
            "mem_mb": round(float(self.mem_mb), 2) if self.mem_mb is not None else "",
            "post_count": self.post_count or 0,
            "n_used": self.n_used or 0,
            "is_reference": int(self.is_reference or False),
            "pipeline": self.pipeline
        }
