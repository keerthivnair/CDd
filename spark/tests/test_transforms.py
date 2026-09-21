import json
import os
import sys

import pytest
from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transforms as T  # noqa: E402

FMT = "yyyy-MM-dd HH:mm:ss"


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder.master("local[2]").appName("task2-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()


def kafka_like(spark, msgs):
    """Mimic the Kafka source: `value` is binary."""
    return spark.createDataFrame([(m.encode("utf-8"),) for m in msgs], "value binary")


def msg(pid, ts, text, source="twitter"):
    return json.dumps({"post_id": pid, "timestamp": ts, "text": text, "source": source})


def cleaned(spark, msgs):
    return T.clean_records(T.parse_kafka_value(kafka_like(spark, msgs)))


def test_1_json_parsing(spark):
    df = cleaned(spark, [msg("tweet_1", "2020-04-19", "COVID is spreading https://example.com")])
    r = df.select("post_id", "source", "text", F.date_format("event_time", FMT).alias("et")).first()
    assert r["post_id"] == "tweet_1" and r["source"] == "twitter"
    assert r["et"] == "2020-04-19 00:00:00"          # date-only stays midnight, nothing invented
    assert r["text"] == "covid is spreading"


def test_2_url_mention_hashtag_cleaning(spark):
    df = cleaned(spark, [
        msg("a", "2020-04-19", "COVID is spreading https://example.com @someone"),
        msg("b", "2020-04-19", "Stay safe #COVID19 and #lockdown!!!"),
    ]).orderBy("post_id")
    assert [r["text"] for r in df.collect()] == ["covid is spreading", "stay safe covid19 and lockdown!"]


def test_3_empty_after_cleaning_removed(spark):
    df = cleaned(spark, [msg("a", "2020-04-19", "https://t.co/x @user"), msg("b", "2020-04-19", "real text")])
    assert [r["post_id"] for r in T.filter_valid(df).collect()] == ["b"]
    assert T.quality_counts(df)["empty_after_cleaning"] == 1


def test_5_invalid_timestamp_detected_not_invented(spark):
    df = cleaned(spark, [msg("a", "not-a-date", "hello"), msg("b", "2020-04-19", "hello")])
    assert df.filter(F.col("post_id") == "a").first()["event_time"] is None
    assert [r["post_id"] for r in T.filter_valid(df).collect()] == ["b"]
    assert T.quality_counts(df)["invalid_timestamp"] == 1


def test_malformed_json_counted(spark):
    df = cleaned(spark, ["{not json", msg("b", "2020-04-19", "ok")])
    assert T.quality_counts(df)["malformed_json"] == 1


def test_duration_parser():
    assert T.duration_to_seconds("15 minutes") == 900
    assert T.duration_to_seconds("1 day") == 86400
    with pytest.raises(ValueError):
        T.duration_to_seconds("soon")


def test_4_and_6_dedup_and_window_assignment_streaming(spark, tmp_path):
    """Real streaming query (file source -> watermark -> dedup -> windows -> memory sink)."""
    src = tmp_path / "in"
    src.mkdir()
    rows = [
        {"post_id": "tweet_1", "timestamp": "2020-04-19 14:02:00", "text": "First one https://x.co", "source": "twitter"},
        {"post_id": "tweet_1", "timestamp": "2020-04-19 14:02:00", "text": "First one https://x.co", "source": "twitter"},  # duplicate
        {"post_id": "tweet_2", "timestamp": "2020-04-19 14:07:00", "text": "Second", "source": "twitter"},
        {"post_id": "tweet_3", "timestamp": "2020-04-19 14:16:00", "text": "Third", "source": "twitter"},
        {"post_id": "tweet_4", "timestamp": "2020-04-19 14:20:00", "text": "@a https://b.co", "source": "twitter"},  # empty -> dropped
        {"post_id": "tweet_9", "timestamp": "2020-04-20 00:00:00", "text": "sentinel", "source": "twitter"},  # advances watermark
    ]
    (src / "a.json").write_text("\n".join(json.dumps(r) for r in rows))

    raw = spark.readStream.schema(T.MESSAGE_SCHEMA).json(str(src))
    windows = T.build_windows(T.filter_valid(T.clean_records(raw.select(
        F.col("post_id"), F.col("timestamp"), F.col("text"), F.col("source")))), "15 minutes", "10 minutes")

    q = (T.to_output_json_frame(windows).writeStream.format("memory").queryName("win_test")
         .outputMode("append").option("checkpointLocation", str(tmp_path / "ckpt")).start())
    q.processAllAvailable()
    q.stop()

    out = {r["window_start"]: r for r in spark.table("win_test").collect()}
    w1, w2 = out["2020-04-19T14:00:00"], out["2020-04-19T14:15:00"]
    assert w1["window_end"] == "2020-04-19T14:15:00"
    assert list(w1["post_ids"]) == ["tweet_1", "tweet_2"]           # duplicate removed
    assert list(w1["texts"]) == ["first one", "second"]
    assert list(w2["post_ids"]) == ["tweet_3"]                     # empty tweet_4 removed
    assert w1["post_count"] == 2 and w2["post_count"] == 1
    assert w1["window_id"] == 1_587_304_800 // 900                # floor(epoch(window_start)/900)
    assert w2["window_id"] == w1["window_id"] + 1                   # consecutive, deterministic ids


def test_rt_prefix_and_html_entity(spark):
    df = cleaned(spark, [
        msg("a", "2020-04-19", "RT @GlblCtzn: Vaccines &amp; masks work"),
        msg("b", "2020-04-19", "RTX 3080 prices during covid"),
    ]).orderBy("post_id")
    assert [r["text"] for r in df.collect()] == ["vaccines & masks work", "rtx 3080 prices during covid"]


def test_daily_windows_close_when_next_day_arrives(spark, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    rows = [
        {"post_id": "t1", "timestamp": "2020-04-19", "text": "one", "source": "twitter"},
        {"post_id": "t2", "timestamp": "2020-04-19", "text": "two", "source": "twitter"},
        {"post_id": "t3", "timestamp": "2020-04-20", "text": "three", "source": "twitter"},
    ]
    (src / "a.json").write_text("\n".join(json.dumps(r) for r in rows))
    raw = spark.readStream.schema(T.MESSAGE_SCHEMA).json(str(src))
    windows = T.build_windows(T.filter_valid(T.clean_records(raw)), "1 day", "0 seconds")
    q = (T.to_output_json_frame(windows).writeStream.format("memory").queryName("daily_test")
         .outputMode("append").option("checkpointLocation", str(tmp_path / "ckpt")).start())
    q.processAllAvailable()
    q.stop()
    out = spark.table("daily_test").collect()
    assert [r["window_start"] for r in out] == ["2020-04-19T00:00:00"]   # day 1 closed by day 2; day 2 still open
    assert out[0]["post_count"] == 2
