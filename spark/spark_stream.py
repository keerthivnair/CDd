"""Task 2: Kafka -> Spark Structured Streaming -> cleaned event-time tweet windows.

Run from the repository root:
    python spark/spark_stream.py --config spark/config.yaml
"""
import argparse
import logging
import os
import sys

import yaml
import pyspark
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.streaming import StreamingQueryListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transforms as T  # noqa: E402

log = logging.getLogger("spark_stream")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def kafka_package(cfg: dict) -> str:
    if cfg["spark"].get("kafka_package"):
        return cfg["spark"]["kafka_package"]
    v = pyspark.__version__
    scala = "2.13" if int(v.split(".")[0]) >= 4 else "2.12"
    return f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{v}"


def create_spark(cfg: dict) -> SparkSession:
    s = cfg["spark"]
    spark = (
        SparkSession.builder.appName(s["app_name"])
        .master(s.get("master", "local[*]"))
        .config("spark.jars.packages", kafka_package(cfg))
        .config("spark.sql.shuffle.partitions", str(s.get("shuffle_partitions", 8)))
        .config("spark.sql.session.timeZone", s.get("session_timezone", "UTC"))
        # Strict, predictable timestamp parsing: unparseable -> NULL, never an exception or guess.
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


class ProgressLogger(StreamingQueryListener):
    """Logs per-micro-batch progress: input rows, dedup/late-data metrics from the state operators."""

    def onQueryStarted(self, event):
        log.info("Query started: %s", event.name or event.id)

    def onQueryProgress(self, event):
        p = event.progress
        log.info("batch=%s input_rows=%s", p.batchId, p.numInputRows)
        for op in p.stateOperators:
            log.info(
                "  state[%s] rows_total=%s dropped_by_watermark=%s custom=%s",
                getattr(op, "operatorName", "?"),
                getattr(op, "numRowsTotal", "?"),
                getattr(op, "numRowsDroppedByWatermark", "?"),
                dict(getattr(op, "customMetrics", {}) or {}),  # includes duplicate rows dropped
            )

    def onQueryIdle(self, event):
        pass

    def onQueryTerminated(self, event):
        log.info("Query terminated: %s", event.id)


def read_kafka(spark: SparkSession, cfg: dict):
    k = cfg["kafka"]
    reader = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", k["bootstrap_servers"])
        .option("subscribe", k["topic"])
        .option("startingOffsets", k.get("starting_offsets", "earliest"))
        .option("failOnDataLoss", "false")
        .option("groupIdPrefix", k.get("consumer_group_prefix", "cdd-spark"))
    )
    if k.get("max_offsets_per_trigger"):
        reader = reader.option("maxOffsetsPerTrigger", str(k["max_offsets_per_trigger"]))
    log.info("Subscribing to topic '%s' on %s", k["topic"], k["bootstrap_servers"])
    return reader.load()


def make_window_sink(cfg: dict):
    out = cfg["output"]
    sink = out.get("sink", "both")
    max_texts = int(out.get("console_max_texts", 5))
    jsonl_path = resolve(out["jsonl_path"])

    def write_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        frame = T.to_output_json_frame(batch_df).persist()
        try:
            if sink in ("console", "both"):
                # Only windows closed in THIS micro-batch, texts truncated -> small collect.
                preview = frame.select(
                    "window_id", "window_start", "window_end", "post_count",
                    F.slice("texts", 1, max_texts).alias("texts"),
                ).orderBy("window_id").collect()
                for r in preview:
                    print("-" * 60)
                    print(f"Window ID: {r['window_id']}")
                    print(f"Start: {r['window_start']}")
                    print(f"End:   {r['window_end']}")
                    print(f"Posts: {r['post_count']}\n\nTexts (first {max_texts}):")
                    for i, t in enumerate(r["texts"], 1):
                        print(f"{i}. {t}")
                    print("-" * 60)
            if sink in ("jsonl", "both"):
                # One folder per batch, overwrite => idempotent if a batch is retried after a failure.
                frame.orderBy("window_id").coalesce(1).write.mode("overwrite").json(
                    os.path.join(jsonl_path, f"batch_id={batch_id}")
                )
            for r in frame.select("window_id", "post_count").collect():
                log.info("window produced: id=%s posts=%s", r["window_id"], r["post_count"])
        finally:
            frame.unpersist()

    return write_batch


def start_quality_monitor(parsed_stream, cfg: dict):
    """Optional second query (re-reads Kafka): logs what the cleaning stage drops per batch."""
    clean = T.clean_records(parsed_stream)

    def log_counts(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        c = T.quality_counts(batch_df)
        log.info(
            "quality batch=%s total=%s malformed_json=%s invalid_timestamp=%s empty_after_cleaning=%s",
            batch_id, c["total"], c["malformed_json"], c["invalid_timestamp"], c["empty_after_cleaning"],
        )

    return (
        clean.writeStream.foreachBatch(log_counts)
        .option("checkpointLocation", resolve(cfg["output"]["checkpoint_location"]) + "_quality")
        .trigger(processingTime=cfg["streaming"]["trigger_interval"])
        .start()
    )


def main():
    parser = argparse.ArgumentParser(description="Task 2: Kafka -> Spark Structured Streaming")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--window-duration", help="Override streaming.window_duration, e.g. '30 minutes'")
    parser.add_argument("--quality-report", action="store_true", help="Also log invalid/empty record counts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("py4j").setLevel(logging.WARNING)   # silence "Received command c" noise
    cfg = load_config(args.config)
    if args.window_duration:
        cfg["streaming"]["window_duration"] = args.window_duration
    st = cfg["streaming"]

    spark = create_spark(cfg)
    log.info("Spark %s application started (window=%s, watermark=%s)",
             spark.version, st["window_duration"], st["watermark_duration"])
    spark.streams.addListener(ProgressLogger())

    parsed = T.parse_kafka_value(read_kafka(spark, cfg))
    windows = T.build_windows(
        T.filter_valid(T.clean_records(parsed)), st["window_duration"], st["watermark_duration"]
    )

    if args.quality_report:
        start_quality_monitor(parsed, cfg)

    query = (
        windows.writeStream.outputMode(cfg["output"].get("output_mode", "append"))
        .foreachBatch(make_window_sink(cfg))
        .option("checkpointLocation", resolve(cfg["output"]["checkpoint_location"]))
        .trigger(processingTime=st["trigger_interval"])
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
