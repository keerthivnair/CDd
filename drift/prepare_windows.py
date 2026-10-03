"""Utility to generate windows_all.jsonl directly from Kaggle CSVs.

Applies the exact same cleaning and windowing rules as Task 2 (spark/transforms.py).
Run from the repository root:
    python drift/prepare_windows.py
"""
import glob
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_windows")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_DIR = os.path.join(PROJECT_ROOT, "ingestion", "data", "kaggle")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "windows_all.jsonl")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEATED_PUNCT_RE = re.compile(r"([!?.])\1+")
RT_PREFIX_RE = re.compile(r"^\s*rt\b\s*:?\s*")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    c = text.lower()
    c = URL_RE.sub(" ", c)
    c = MENTION_RE.sub(" ", c)
    c = RT_PREFIX_RE.sub("", c)
    c = c.replace("&amp;", "&")
    c = HASHTAG_RE.sub(r"\1", c)
    c = REPEATED_PUNCT_RE.sub(r"\1", c)
    c = re.sub(r"\s+", " ", c)
    return c.strip()


def main():
    csv_files = sorted(glob.glob(os.path.join(KAGGLE_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {KAGGLE_DIR}")

    log.info("Reading %d Kaggle CSV files...", len(csv_files))
    all_dfs = []
    row_idx = 0
    for p in csv_files:
        df = pd.read_csv(p, usecols=["created_at", "original_text"])
        df["post_id"] = [f"tweet_{i}" for i in range(row_idx, row_idx + len(df))]
        row_idx += len(df)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.dropna(subset=["created_at", "original_text"])
    combined["event_time"] = pd.to_datetime(combined["created_at"], errors="coerce")
    combined = combined.dropna(subset=["event_time"])

    log.info("Cleaning %d tweet texts...", len(combined))
    combined["text"] = combined["original_text"].apply(clean_text)
    combined = combined[combined["text"].str.len() > 0]

    # Deduplicate post_id and event_time (same as Spark watermark dedup)
    combined = combined.drop_duplicates(subset=["post_id", "event_time"])
    combined = combined.sort_values("event_time")

    log.info("Building 1-day windows...")
    iso = "%Y-%m-%dT%H:%M:%S"
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for day, group in combined.groupby(combined["event_time"].dt.date):
            start_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = start_dt + timedelta(days=1)
            window_id = int(start_dt.timestamp() // 86400)
            texts = group["text"].tolist()
            post_ids = group["post_id"].tolist()
            timestamps = [start_dt.strftime(iso)] * len(texts)

            row = {
                "window_id": window_id,
                "window_start": start_dt.strftime(iso),
                "window_end": end_dt.strftime(iso),
                "texts": texts,
                "timestamps": timestamps,
                "post_ids": post_ids,
                "post_count": len(texts),
            }
            out.write(json.dumps(row) + "\n")
            written += 1

    log.info("Done: wrote %d windows (%d tweets) to %s", written, len(combined), OUTPUT_PATH)


if __name__ == "__main__":
    main()
