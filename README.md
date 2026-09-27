# Concept Drift Detection in Social Media Streams

A streaming data pipeline for detecting concept drift in unlabeled social media text, using a historical tweet dataset replayed through Kafka to simulate a live stream.

## Motivation

Public discourse around an evolving topic shifts over time — the language, sentiment, and dominant sub-topics people use to discuss it change as events unfold. Concept drift detection aims to identify *when* and *how much* this shift occurs, without relying on manual labeling. This is useful for monitoring public sentiment during ongoing events, detecting emerging narratives, and flagging when a downstream model trained on older data may no longer reflect current language patterns.

## Architecture

```
Social media dataset (historical tweets)
        |
Ingestion: Kafka producer                    [implemented]
        |  validates and standardizes records, replays them to Kafka
        |  in original chronological order, simulating a live stream
        v
Kafka topic (JSON tweet events)
        |
Stream processing: Spark Structured Streaming     [implemented]
        |  parses records, derives event time from each record's own
        |  timestamp, cleans text, removes empty/duplicate records,
        |  aggregates into fixed event-time windows
        v
Cleaned, windowed text (JSON Lines)
        |
Embedding + drift scoring                         [planned]
        |  sentence embeddings per window; distributional distance
        |  between window embeddings (e.g. Frechet distance,
        |  covariance-based drift metrics) to score drift over time
        v
Monitoring / visualization                         [planned]
        |  time-series store and dashboard of drift scores
```

## Repository layout

| Path | Component | Status |
|---|---|---|
| `ingestion/` | Kafka producer: loads, validates, and replays the dataset | Implemented |
| `spark/` | Spark Structured Streaming: cleaning, deduplication, event-time windowing | Implemented |
| — | Embedding generation and drift scoring | Planned |
| — | Metrics store and dashboard | Planned |

Each component has its own README with setup, configuration, and usage details.

## Design principles

- **Event time over processing time.** Windows are built from each record's own timestamp, not from when it was ingested or processed — important when replaying historical data, where processing order does not reflect the original temporal order.
- **No labels required.** Drift is measured on the distribution of the data itself (via embeddings), not against ground-truth labels, so the pipeline works on arbitrary unlabeled text streams.
- **Streaming-native processing.** Data flows through Kafka and Spark Structured Streaming rather than being loaded and processed as a single static batch, so the pipeline reflects how a live deployment would operate.
- **No fabricated precision.** Where the source data has coarser time granularity than an analysis step would prefer, the pipeline adapts window sizes accordingly rather than inventing timestamp precision that isn't present in the data.

## Getting started

Each component is independently runnable and tested. See:
- `ingestion/README.md` for producing data to Kafka.
- `spark/README.md` for running the streaming windowing pipeline, its configuration, and the output format consumed by the next stage.

## Status

Data ingestion and stream windowing are implemented and tested end-to-end on a multi-hundred-thousand-record dataset. Embedding-based drift scoring and the monitoring layer are the next stages of the pipeline.
