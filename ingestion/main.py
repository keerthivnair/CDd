import argparse
import yaml
from loader import load_dataset
from validator import validate
from producer import standardize, replay_to_kafka

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="COVID-19 Twitter Dataset Ingestion")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--no-delay", action="store_true", help="Disable replay delay")
    parser.add_argument("--replay-speed", type=float, help="Replay speed multiplier")
    parser.add_argument("--max-records", type=int, help="Maximum number of records to process")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    

    if args.no_delay:
        config['no_delay'] = True
    if args.replay_speed:
        config['replay_speed'] = args.replay_speed
        
    dataset_path = config.get('dataset_path', 'ingestion/data/kaggle/')
    
    # Make path robust: if it's relative, resolve it relative to the project root
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.isabs(dataset_path):
        # If the user runs from inside `ingestion` but the config says `ingestion/data/...`, 
        # it's easiest to always resolve it from the project root
        dataset_path = os.path.join(project_root, dataset_path)
    
    print(f"Loading dataset from: {dataset_path}")
    raw_records = load_dataset(dataset_path)
    
    valid_records = validate(raw_records)
    

    std_records = [standardize(r) for r in valid_records]
    
    replay_to_kafka(std_records, config, args.max_records)

if __name__ == "__main__":
    main()
