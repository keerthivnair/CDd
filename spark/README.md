# Task 2 - Spark Structured Streaming

**What this folder does:** reads the tweet stream that Task 1 publishes to Kafka, cleans it, and groups it into event-time windows of cleaned tweets. The output is what Task 3 (embeddings and drift detection) consumes.

```
Kaggle COVID-19 Twitter Dataset
        |
Task 1: ingestion/ producer  ->  Kafka topic `social-media-stream`
        |
Task 2: Spark Structured Streaming (this folder)
        |  1. read Kafka as a stream (readStream)
        |  2. parse JSON with an explicit schema
        |  3. convert `timestamp` to event_time
        |  4. clean text
        |  5. drop empty / invalid records
        |  6. watermark + drop duplicate post_id
        |  7. fixed event-time windows (window_id, start, end, texts)
        v
cleaned tweet windows (console preview + JSONL)  ->  Task 3
```

Task 1 was not changed. Input per Kafka message:
`{"post_id": "tweet_123", "timestamp": "2020-04-19", "text": "...", "source": "twitter"}`

## What was implemented

| Step | What it does |
|---|---|
| Kafka source | `spark.readStream` on topic `social-media-stream` (`localhost:9092`), batches capped by `max_offsets_per_trigger`. No Python consumer, no Pandas, no `collect()` on the stream. |
| JSON parsing | Kafka binary `value` -> string -> `from_json` with a fixed schema (`post_id`, `timestamp`, `text`, `source`). No schema inference. Malformed JSON gives NULL fields and is dropped and counted. |
| Event time | `event_time` is parsed from the tweet's own `timestamp` (never from Kafka's or Spark's clock). Unparseable values become NULL and are dropped, not replaced with an invented time. |
| Text cleaning | Spark-native functions only (no UDFs): lower-case; remove URLs; remove `@mentions`; remove the leading `RT` marker (word-boundary safe, so `RTX 3080` stays); `&amp;` -> `&`; `#tag` -> `tag`; repeated `!?.` collapsed; whitespace normalized. No stopwords, stemming or tokenization, so meaning is kept for the Sentence Transformer. |
| Empty removal | After cleaning, NULL or blank texts are dropped (for example a tweet that was only a URL and a mention). |
| Deduplication | `dropDuplicates(["post_id", "event_time"])` after the watermark (Spark's supported streaming dedup; `event_time` is in the key so the watermark can expire the state). |
| Watermark | Configurable; `0 seconds` by default (see "Design decisions"). |
| Windows | `window(event_time, window_duration)`, default `1 day`, configurable. Per window: `window_id`, `window_start`, `window_end`, `texts`, `timestamps`, `post_ids`, `post_count`. |
| `window_id` | `floor(epoch_seconds(window_start) / window_length_seconds)`. Deterministic; for daily windows `18371` = 2020-04-19. |
| Sinks | Console preview (first N texts per window) and JSONL files, via `foreachBatch`. Each batch is written to its own folder with overwrite, so a retried batch does not duplicate data. |
| Checkpointing | Structured Streaming checkpoint at `checkpoints/spark_stream` (configurable). |
| Logging | Spark start, topic subscription, rows per micro-batch, duplicates dropped and rows dropped by the watermark, windows produced with post counts. With `--quality-report`, also malformed / invalid-timestamp / empty-after-cleaning counts per batch. |

## Files

| File | Purpose |
|---|---|
| `spark_stream.py` | The streaming application (Kafka source, sinks, logging, CLI) |
| `transforms.py` | The transformations as pure Spark functions (unit-tested) |
| `config.yaml` | All settings |
| `requirements.txt` | `pyspark==3.5.3`, `pyyaml`, `pytest` |
| `tests/test_transforms.py` | 9 tests, no Kafka needed |

## How to run (from the repository root)

Requirements: Java 17 or 21, Python 3.9-3.12 (PySpark 3.5.3 does not support 3.13+). The Kafka connector for the installed PySpark is downloaded automatically on the first run. On Windows, run the Spark step inside WSL to avoid the `winutils`/`HADOOP_HOME` problem (and `unset HADOOP_HOME` if a separate Hadoop install exists).

```bash
pip install -r spark/requirements.txt -r ingestion/requirements.txt
```

1. **Start Kafka** on `localhost:9092`, for example:
   ```bash
   docker run -d --name kafka -p 9092:9092 apache/kafka:3.9.0
   docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --topic social-media-stream --bootstrap-server localhost:9092
   ```
2. **Start Spark** (terminal 2):
   ```bash
   python spark/spark_stream.py --config spark/config.yaml --quality-report
   ```
3. **Run the existing producer** (terminal 3). Small test first, then the full run without `--max-records`:
   ```bash
   cd ingestion
   python main.py --config config.yaml --no-delay --max-records 20000
   ```

Stop Spark with Ctrl+C once the `batch=N` log lines stop. To start from scratch, recreate the Kafka topic and run `rm -rf checkpoints checkpoints_quality output`, otherwise Spark resumes from old offsets.

**Tests:** `cd spark && python -m pytest tests -q`. They cover JSON parsing, URL/mention/hashtag/RT cleaning, empty-text removal, invalid-timestamp detection, malformed JSON, duplicate removal and window assignment (a real streaming query using a file source), and daily windows closing when the next day arrives.

## Configuration (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `kafka.bootstrap_servers` / `topic` | `localhost:9092` / `social-media-stream` | Kafka source |
| `kafka.consumer_group_prefix` | `cdd-spark` | Prefix for Spark's consumer group ids (offsets are tracked in the checkpoint) |
| `kafka.starting_offsets` | `earliest` | Replay from the start |
| `kafka.max_offsets_per_trigger` | `20000` | Messages per micro-batch |
| `streaming.window_duration` | `1 day` | Event-time window size (`--window-duration` overrides it) |
| `streaming.watermark_duration` | `0 seconds` | Watermark delay |
| `output.checkpoint_location` | `checkpoints/spark_stream` | Checkpoint folder |
| `output.output_mode` | `append` | Each window is emitted once, after it closes |
| `output.sink` | `both` | `console`, `jsonl` or `both` |
| `output.jsonl_path` | `output/windows` | JSONL output folder |

Relative paths resolve from the repository root. If `/mnt/<drive>` in WSL is slow, use Linux-side absolute paths for the checkpoint and output.

## Design decisions

- **Event time, not ingestion time.** Historical tweets are replayed through Kafka, so their original dates define the stream, not the time Kafka or Spark received them.
- **Late data and the watermark.** *Late data* is a record older than the newest event time already seen, by more than the allowed delay. The *watermark* (max event time seen minus `watermark_duration`) tells Spark when a window is complete: once it passes a window's end, the window is emitted and its state freed, and later records for it are dropped. It also bounds the dedup state. It is an implementation parameter, not a research result.
- **Why `0 seconds`.** The data has dates only, so a day's tweets all sit at 00:00. With a 10-minute watermark a day would close only when the day *after next* arrived. With `0 seconds` it closes as soon as the next day arrives. Increase it if you switch to data with real times and out-of-order arrival.
- **Daily windows, not 15/30 minutes.** The Kaggle dataset has no time of day, so 15/30-minute windows would put a whole day in one window at midnight. Nothing was invented to make them work. The window size stays configurable.
- **Dedup on `post_id`.** Task 1 already removes duplicate text+date rows and creates unique `post_id`s, so this stage mainly guards against re-published messages.

## Result on the full dataset (411,885 records)

- 153 windows emitted (154 days minus the still-open last day), all unique, window ids 18371-18804.
- 410,455 posts. This equals the 410,461 posts before the last day minus 6 tweets that were empty after cleaning (checked independently against the CSVs).
- In a small test, a second copy of 20,000 messages was fully removed by the deduplication stage.

## Known limitations

1. **Date-only timestamps** (see above): windows are daily. The parser also accepts `yyyy-MM-dd HH:mm:ss` and ISO `T` timestamps, so finer data would work without code changes.
2. **The final window is never emitted.** In append mode a window closes only when newer data arrives, so the last day (2021-06-27, 1,424 posts) stays open.
3. **Late records are dropped** by the watermark. The producer replays in date order, so none are expected.
4. **Retweets and truncated text.** About 59% of tweets are retweets, and many end with a truncated `...` from the source. Left as is.
5. **Window contents live in Spark state until the window closes** (up to about 7,000 tweets per day here), fine for a local run.

## Output for Task 3

JSONL at `output/windows/batch_id=<n>/part-*.json`, one line per window:

```json
{"window_id": 18371, "window_start": "2020-04-19T00:00:00", "window_end": "2020-04-20T00:00:00",
 "texts": ["covid-19: ncdc trains police officers in anambra", "..."],
 "timestamps": ["2020-04-19T00:00:00", "..."], "post_ids": ["tweet_12", "..."], "post_count": 339}
```

- `texts` is ordered by event time, then `post_id`, and is ready for the Sentence Transformer.
- `timestamps` and `post_ids` line up index by index with `texts` (for debugging).
- Timestamps are ISO-8601 strings in UTC.
- Batch folders sort as text (`batch_id=10` before `batch_id=2`), so sort by `window_id` when reading:
  ```python
  import glob, json
  windows = []
  for p in glob.glob("output/windows/batch_id=*/part-*.json"):
      windows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
  windows.sort(key=lambda w: w["window_id"])
  ```