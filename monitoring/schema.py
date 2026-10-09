"""
Data schemas for Task 4: Drift, Explainability & System Metrics Monitoring.
Implements the full unified schema combining DriftLens, MCD-DD, agreement classification,
and host performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union, Dict, Any


def classify_agreement(is_reference: bool, dl_alarm: bool, mcd_alarm: Optional[bool]) -> str:
    """Computes exact detector agreement from actual alarm fields."""
    if is_reference:
        return "reference"
    if mcd_alarm is None:
        return "mcddd_missing"
    dl = bool(dl_alarm)
    mcd = bool(mcd_alarm)
    if dl and mcd:
        return "both"
    elif dl and not mcd:
        return "driftlens_only"
    elif not dl and mcd:
        return "mcddd_only"
    else:
        return "neither"


AGREEMENT_CODES = {
    "reference": 0,
    "neither": 1,
    "driftlens_only": 2,
    "mcddd_only": 3,
    "both": 4,
    "mcddd_missing": 5,
}


@dataclass
class WindowDriftResult:
    """
    Represents the output of a single streaming window evaluation.
    Directly consumes real detector output fields (DriftLens & MCD-DD)
    and supplements with agreement classification and host performance metrics.
    """
    timestamp: Union[datetime, str, int, float]
    window_id: Union[str, int]
    driftlens_score: float
    driftlens_alarm: bool
    processing_latency_ms: float
    throughput_posts_per_sec: float
    driftlens_threshold: Optional[float] = None
    driftlens_ratio: Optional[float] = None
    mcddd_score: Optional[float] = None
    mcddd_threshold: Optional[float] = None
    mcddd_alarm: Optional[bool] = None
    mcddd_ratio: Optional[float] = None
    mcddd_lag: Optional[int] = None
    agreement: str = "neither"
    agreement_code: int = 1
    n_used: Optional[int] = 0
    post_count: Optional[int] = 0
    is_reference: Optional[bool] = False
    cpu_pct: Optional[float] = None
    mem_mb: Optional[float] = None
    pipeline: str = "cdd-stream"

    @classmethod
    def from_task3_dict(
        cls,
        data: Dict[str, Any],
        default_latency: float = 0.0,
        default_throughput: float = 0.0
    ) -> "WindowDriftResult":
        """
        Validates and constructs a WindowDriftResult from detector dictionary output.
        """
        if "window_id" not in data or data["window_id"] is None:
            raise ValueError("Detector result missing required field: 'window_id'")

        if "driftlens_score" not in data or data["driftlens_score"] is None:
            raise ValueError(f"Result for window {data.get('window_id')} missing required field: 'driftlens_score'")

        try:
            dl_score = float(data["driftlens_score"])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid non-numeric driftlens_score: {data.get('driftlens_score')}")

        if "driftlens_alarm" not in data or data["driftlens_alarm"] is None:
            raise ValueError(f"Result for window {data.get('window_id')} missing required field: 'driftlens_alarm'")
        dl_alarm = bool(data["driftlens_alarm"])

        ts_val = data.get("window_start") or data.get("timestamp")
        if not ts_val:
            raise ValueError(f"Result for window {data.get('window_id')} missing timestamp / window_start")

        # Thresholds
        thr = data.get("threshold", data.get("driftlens_threshold"))
        dl_threshold = float(thr) if thr is not None else None

        # MCD-DD fields
        mcd_score_val = data.get("mcddd_score")
        mcd_score = float(mcd_score_val) if mcd_score_val is not None else None

        mcd_thr_val = data.get("mcddd_threshold")
        mcd_threshold = float(mcd_thr_val) if mcd_thr_val is not None else None

        mcd_alarm_val = data.get("mcddd_alarm")
        mcd_alarm = bool(mcd_alarm_val) if mcd_alarm_val is not None else None

        mcd_lag = int(data["mcddd_lag"]) if data.get("mcddd_lag") is not None else None

        # Ratios (score / threshold)
        dl_ratio = round(dl_score / dl_threshold, 4) if (dl_threshold and dl_threshold > 0) else None
        mcd_ratio = (
            round(mcd_score / mcd_threshold, 4)
            if (mcd_score is not None and mcd_threshold and mcd_threshold > 0)
            else None
        )

        is_ref = bool(data.get("is_reference", False))
        agreement = data.get("agreement") or classify_agreement(is_ref, dl_alarm, mcd_alarm)
        agreement_code = AGREEMENT_CODES.get(agreement, 1)

        post_count = int(data.get("n_posts", data.get("post_count", 0)))
        n_used = int(data.get("n_used", 0))

        latency = float(data.get("processing_latency_ms", default_latency))
        throughput = float(data.get("throughput_posts_per_sec", default_throughput))

        if throughput <= 0.0 and latency > 0.0 and post_count > 0:
            throughput = round(post_count / (latency / 1000.0), 2)

        return cls(
            timestamp=ts_val,
            window_id=data["window_id"],
            driftlens_score=dl_score,
            driftlens_alarm=dl_alarm,
            driftlens_threshold=dl_threshold,
            driftlens_ratio=dl_ratio,
            mcddd_score=mcd_score,
            mcddd_threshold=mcd_threshold,
            mcddd_alarm=mcd_alarm,
            mcddd_ratio=mcd_ratio,
            mcddd_lag=mcd_lag,
            agreement=agreement,
            agreement_code=agreement_code,
            processing_latency_ms=latency,
            throughput_posts_per_sec=throughput,
            post_count=post_count,
            n_used=n_used,
            is_reference=is_ref,
            cpu_pct=float(data["cpu_pct"]) if data.get("cpu_pct") is not None else None,
            mem_mb=float(data["mem_mb"]) if data.get("mem_mb") is not None else None,
            pipeline=str(data.get("pipeline", "cdd-stream")),
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
            f"agreement={self.agreement}",
            f"driftlens_alarm_state={'drift' if self.driftlens_alarm else 'normal'}",
            f"is_reference={'true' if self.is_reference else 'false'}",
        ]
        if self.mcddd_alarm is not None:
            tags.append(f"mcddd_alarm_state={'drift' if self.mcddd_alarm else 'normal'}")

        tag_str = ",".join(tags)

        fields = [
            f"driftlens_score={float(self.driftlens_score)}",
            f"driftlens_alarm={1 if self.driftlens_alarm else 0}i",
            f"agreement_code={int(self.agreement_code)}i",
            f"processing_latency_ms={float(self.processing_latency_ms)}",
            f"throughput_posts_per_sec={float(self.throughput_posts_per_sec)}",
        ]
        if self.driftlens_threshold is not None:
            fields.append(f"driftlens_threshold={float(self.driftlens_threshold)}")
        if self.driftlens_ratio is not None:
            fields.append(f"driftlens_ratio={float(self.driftlens_ratio)}")

        if self.mcddd_score is not None:
            fields.append(f"mcddd_score={float(self.mcddd_score)}")
        if self.mcddd_threshold is not None:
            fields.append(f"mcddd_threshold={float(self.mcddd_threshold)}")
        if self.mcddd_alarm is not None:
            fields.append(f"mcddd_alarm={1 if self.mcddd_alarm else 0}i")
        if self.mcddd_ratio is not None:
            fields.append(f"mcddd_ratio={float(self.mcddd_ratio)}")
        if self.mcddd_lag is not None:
            fields.append(f"mcddd_lag={int(self.mcddd_lag)}i")

        if self.cpu_pct is not None:
            fields.append(f"cpu_pct={float(self.cpu_pct)}")
        if self.mem_mb is not None:
            fields.append(f"mem_mb={float(self.mem_mb)}")
        if self.post_count is not None:
            fields.append(f"post_count={int(self.post_count)}i")
            fields.append(f"n_posts={int(self.post_count)}i")
        if self.n_used is not None:
            fields.append(f"n_used={int(self.n_used)}i")

        field_str = ",".join(fields)
        ts_sec = int(self.get_datetime().timestamp())
        return f"{measurement},{tag_str} {field_str} {ts_sec}"

    def to_flat_dict(self) -> Dict[str, Any]:
        """Converts record to a dictionary suitable for CSV serialization."""
        return {
            "timestamp": self.get_datetime().isoformat(),
            "window_id": str(self.window_id),
            "driftlens_score": round(float(self.driftlens_score), 6),
            "driftlens_threshold": round(float(self.driftlens_threshold), 6) if self.driftlens_threshold is not None else "",
            "driftlens_alarm": int(self.driftlens_alarm),
            "driftlens_ratio": round(float(self.driftlens_ratio), 4) if self.driftlens_ratio is not None else "",
            "mcddd_score": round(float(self.mcddd_score), 6) if self.mcddd_score is not None else "",
            "mcddd_threshold": round(float(self.mcddd_threshold), 6) if self.mcddd_threshold is not None else "",
            "mcddd_alarm": int(self.mcddd_alarm) if self.mcddd_alarm is not None else "",
            "mcddd_ratio": round(float(self.mcddd_ratio), 4) if self.mcddd_ratio is not None else "",
            "agreement": self.agreement,
            "processing_latency_ms": round(float(self.processing_latency_ms), 2),
            "throughput_posts_per_sec": round(float(self.throughput_posts_per_sec), 2),
            "cpu_pct": round(float(self.cpu_pct), 2) if self.cpu_pct is not None else "",
            "mem_mb": round(float(self.mem_mb), 2) if self.mem_mb is not None else "",
            "n_posts": self.post_count or 0,
            "n_used": self.n_used or 0,
            "is_reference": int(self.is_reference or False),
            "pipeline": self.pipeline,
        }
