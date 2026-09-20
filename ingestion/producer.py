import json
import time
from datetime import datetime
from confluent_kafka import Producer

def standardize(record: dict) -> dict:
    """
    Standardize the raw record to the expected downstream format.
    {post_id, timestamp, text, source}
    """
    return {
        "post_id": f"tweet_{record.get('row_index')}",
        "timestamp": str(record.get('created_at')).strip(),
        "text": str(record.get('original_text')).strip(),
        "source": "twitter"
    }

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

def replay_to_kafka(records: list[dict], config: dict, max_records: int = None):
    """
    Sorts records chronologically and publishes them to Kafka.
    """
    if not records:
        print("No records to replay.")
        return

    records.sort(key=lambda x: str(x.get('timestamp')))
    
    if max_records and max_records > 0:
        records = records[:max_records]

    kafka_conf = config.get('kafka', {})
    bootstrap_servers = kafka_conf.get('bootstrap_servers', 'localhost:9092')
    topic = kafka_conf.get('topic', 'social-media-stream')
    
    replay_speed = config.get('replay_speed', 100)
    no_delay = config.get('no_delay', False)

    try:
        producer = Producer({'bootstrap.servers': bootstrap_servers})
    except Exception as e:
        print(f"Failed to connect to Kafka at {bootstrap_servers}: {e}")
        return

    print(f"Starting replay of {len(records)} records to topic '{topic}'")
    
    last_dt = None
    count = 0
    
    for record in records:
        current_ts_str = record['timestamp']
        
        try:
          
            current_dt = datetime.strptime(current_ts_str, '%Y-%m-%d')
        except ValueError:
          
            current_dt = last_dt if last_dt else datetime.now()

        if last_dt is not None and not no_delay:
            gap = (current_dt - last_dt).total_seconds()
            if gap > 0:
                delay = gap / replay_speed
                if delay > 0:
                    time.sleep(delay)
                    

        try:
            producer.produce(
                topic,
                key=record['post_id'].encode('utf-8'),
                value=json.dumps(record).encode('utf-8'),
                callback=delivery_report
            )
        except BufferError:
            producer.poll(1)
            producer.produce(
                topic,
                key=record['post_id'].encode('utf-8'),
                value=json.dumps(record).encode('utf-8'),
                callback=delivery_report
            )

        producer.poll(0)
        
        last_dt = current_dt
        count += 1
        
        if count % 5000 == 0:
            print(f"Published {count} records...")

    producer.flush()
    print(f"Replay completed. Total records published: {count}")
