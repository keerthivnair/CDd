"""Pure DataFrame transformations for Task 2 (no Kafka/IO here, so they are unit-testable).

Every function works on streaming and batch DataFrames alike and uses only
Spark SQL functions (no Python UDFs, no collect()).
"""
import re

from pyspark.sql import DataFrame, functions as F, types as T

# Input contract from Task 1 (ingestion/producer.py -> standardize()).
MESSAGE_SCHEMA = T.StructType([
    T.StructField("post_id", T.StringType()),
    T.StructField("timestamp", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("source", T.StringType()),
])

# Task 1 currently emits date-only strings ("2020-04-19"). The finer formats are accepted
# only so that real time-of-day precision is preserved if a dataset ever provides it.
TIMESTAMP_FORMATS = ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd"]

URL_RE = r"(https?://\S+|www\.\S+)"
MENTION_RE = r"(?<!\w)@\w+"
HASHTAG_RE = r"#(\w+)"
REPEATED_PUNCT_RE = r"([!?.])\1+"
RT_PREFIX_RE = r"^\s*rt\b\s*:?\s*"          # residue of "RT @user:" once the mention is stripped

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}


def duration_to_seconds(duration: str) -> int:
    """'15 minutes' -> 900. Used only to derive a deterministic window_id."""
    m = re.fullmatch(r"\s*(\d+)\s*(second|minute|hour|day|week)s?\s*", duration, re.I)
    if not m:
        raise ValueError(f"Unsupported duration '{duration}' (use e.g. '15 minutes', '1 day')")
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]


def parse_kafka_value(kafka_df: DataFrame) -> DataFrame:
    """Kafka binary `value` -> string -> JSON parsed with an explicit schema (no inference)."""
    return (
        kafka_df.select(F.col("value").cast("string").alias("raw_value"))
        .select("raw_value", F.from_json("raw_value", MESSAGE_SCHEMA).alias("m"))
        .select("raw_value", "m.*")
    )


def add_event_time(df: DataFrame) -> DataFrame:
    """event_time comes from the tweet's own `timestamp` field, never from Kafka/Spark clocks.

    Unparseable values become NULL (they are NOT replaced with a made-up time).
    """
    ts = F.trim(F.col("timestamp"))
    return df.withColumn("event_time", F.coalesce(*[F.to_timestamp(ts, f) for f in TIMESTAMP_FORMATS]))


def clean_text_col(col):
    """Lightweight, meaning-preserving cleaning (kept for a Sentence Transformer later)."""
    c = F.lower(col)
    c = F.regexp_replace(c, URL_RE, " ")                 # URLs first (they may contain '#' or '@')
    c = F.regexp_replace(c, MENTION_RE, " ")             # @user
    c = F.regexp_replace(c, RT_PREFIX_RE, "")            # leading "rt :" (word boundary keeps "rtx 3080")
    c = F.regexp_replace(c, r"&amp;", "&")               # HTML entity left by the Twitter API
    c = F.regexp_replace(c, HASHTAG_RE, "$1")            # #covid -> covid (keeps the word)
    c = F.regexp_replace(c, REPEATED_PUNCT_RE, "$1")     # "!!!" -> "!"
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def clean_records(parsed_df: DataFrame) -> DataFrame:
    return add_event_time(parsed_df).withColumn("text", clean_text_col(F.col("text")))


def valid_condition():
    """A record is usable if it has an id, a parseable event time and non-empty cleaned text."""
    return (
        F.col("post_id").isNotNull()
        & F.col("event_time").isNotNull()
        & F.col("text").isNotNull()
        & (F.length(F.trim(F.col("text"))) > 0)
    )


def filter_valid(clean_df: DataFrame) -> DataFrame:
    return clean_df.filter(valid_condition()).select("post_id", "event_time", "text", "source")


def quality_counts(clean_df: DataFrame) -> dict:
    """Counts of what filter_valid() drops. Intended for one micro-batch (batch DataFrame)."""
    row = clean_df.agg(
        F.count("*").alias("total"),
        F.sum(F.when(F.col("post_id").isNull() & F.col("text").isNull(), 1).otherwise(0)).alias("malformed_json"),
        F.sum(F.when(F.col("text").isNotNull() & F.col("event_time").isNull(), 1).otherwise(0)).alias("invalid_timestamp"),
        F.sum(F.when(F.col("event_time").isNotNull() & (F.length(F.trim(F.col("text"))) == 0), 1).otherwise(0)).alias("empty_after_cleaning"),
    ).first()
    return {k: int(row[k] or 0) for k in ("total", "malformed_json", "invalid_timestamp", "empty_after_cleaning")}


def build_windows(valid_df: DataFrame, window_duration: str, watermark_duration: str) -> DataFrame:
    """Watermark -> dedup on post_id -> fixed event-time windows -> one row per window.

    window_id = floor(epoch_seconds(window_start) / window_length_seconds): deterministic,
    derived only from the window start and the configured window size.
    """
    window_seconds = duration_to_seconds(window_duration)

    deduped = (
        valid_df.withWatermark("event_time", watermark_duration)
        # post_id is the identity; event_time is included so the watermark can expire dedup state.
        .dropDuplicates(["post_id", "event_time"])
    )

    agg = deduped.groupBy(F.window("event_time", window_duration).alias("w")).agg(
        # sort_array on structs orders by event_time, then post_id -> deterministic ordering.
        F.sort_array(F.collect_list(F.struct("event_time", "post_id", "text"))).alias("rows")
    )

    return agg.select(
        F.floor(F.col("w.start").cast("long") / window_seconds).cast("long").alias("window_id"),
        F.col("w.start").alias("window_start"),
        F.col("w.end").alias("window_end"),
        F.col("rows.text").alias("texts"),
        F.col("rows.event_time").alias("timestamps"),
        F.col("rows.post_id").alias("post_ids"),
        F.size("rows").alias("post_count"),
    )


def to_output_json_frame(windows_df: DataFrame) -> DataFrame:
    """Timestamps -> ISO strings so JSONL is easy for Task 3 to read."""
    iso = "yyyy-MM-dd'T'HH:mm:ss"
    return windows_df.select(
        "window_id",
        F.date_format("window_start", iso).alias("window_start"),
        F.date_format("window_end", iso).alias("window_end"),
        "texts",
        F.transform("timestamps", lambda t: F.date_format(t, iso)).alias("timestamps"),
        "post_ids",
        "post_count",
    )
