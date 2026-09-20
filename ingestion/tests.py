import unittest
from validator import validate
from producer import standardize, replay_to_kafka
from loader import load_dataset
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

class TestIngestion(unittest.TestCase):
    
    def test_validation(self):
        records = [
            {"created_at": "2020-04-19", "original_text": "text1", "row_index": "0"},
            {"created_at": "2020-04-19", "original_text": "text1", "row_index": "1"}, # duplicate
            {"created_at": None, "original_text": "text2", "row_index": "2"}, # missing date
            {"created_at": "2020-04-20", "original_text": "", "row_index": "3"}, # missing text
            {"created_at": "2020-04-21", "original_text": "text3", "row_index": "4"}
        ]
        valid = validate(records)
        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]['original_text'], "text1")
        self.assertEqual(valid[1]['original_text'], "text3")

    def test_standardize(self):
        record = {"created_at": "2020-04-19", "original_text": "hello", "row_index": "42", "other": "ignored"}
        std = standardize(record)
        self.assertEqual(std, {
            "post_id": "tweet_42",
            "timestamp": "2020-04-19",
            "text": "hello",
            "source": "twitter"
        })

    def test_sorting_and_timestamps(self):
        records = [
            {"post_id": "2", "timestamp": "2020-04-20", "text": "b", "source": "twitter"},
            {"post_id": "1", "timestamp": "2020-04-19", "text": "a", "source": "twitter"},
            {"post_id": "3", "timestamp": "2020-04-21", "text": "c", "source": "twitter"}
        ]
        
        # Test sorting logic directly
        records.sort(key=lambda x: x.get('timestamp'))
        self.assertEqual(records[0]['post_id'], "1")
        self.assertEqual(records[1]['post_id'], "2")
        self.assertEqual(records[2]['post_id'], "3")
        # Timestamps unchanged
        self.assertEqual(records[0]['timestamp'], "2020-04-19")

    @patch('producer.Producer')
    def test_kafka_producer(self, MockProducer):
        mock_producer = MagicMock()
        MockProducer.return_value = mock_producer
        
        records = [
            {"post_id": "1", "timestamp": "2020-04-19", "text": "a", "source": "twitter"},
        ]
        
        config = {'no_delay': True, 'kafka': {'topic': 'test-topic'}}
        replay_to_kafka(records, config)
        
        mock_producer.produce.assert_called_once()
        args, kwargs = mock_producer.produce.call_args
        
        self.assertEqual(args[0], 'test-topic')
        self.assertEqual(kwargs['key'], b'1')
        val_dict = json.loads(kwargs['value'].decode('utf-8'))
        self.assertEqual(val_dict['text'], 'a')
        self.assertEqual(val_dict['source'], 'twitter')

if __name__ == '__main__':
    unittest.main()
