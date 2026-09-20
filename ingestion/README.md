# COVID-19 Twitter Dataset Ingestion

This module ingests the [Kaggle - Covid-19 Twitter Dataset](https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset) and publishes it to a Kafka topic for downstream consumption.

## 1. Data Ingestion (Producer)
To load the dataset and push all records to Kafka:
```bash
cd ingestion
pip install -r requirements.txt
python main.py --config config.yaml --no-delay
```
This publishes ~411,000 JSON messages to the `social-media-stream` topic on `localhost:9092`.

## 2. Downstream Consumption
To consume the data from Kafka (e.g., via Spark or another service), connect to the `social-media-stream` topic.

**Message Format:**
```json
{
    "post_id": "tweet_123",
    "timestamp": "2020-04-19",
    "text": "Full tweet text here...",
    "source": "twitter"
}
```

**Example Python Consumer:**
```python
from confluent_kafka import Consumer

c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'downstream-group', 
    'auto.offset.reset': 'earliest'
})
c.subscribe(['social-media-stream'])

print("Listening for tweets...")
while True:
    msg = c.poll(1.0)
    if msg and not msg.error():
        print("Received:", msg.value().decode('utf-8'))
```
