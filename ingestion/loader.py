import pandas as pd
import os
import glob

def load_dataset(path: str) -> list[dict]:
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
    elif os.path.isfile(path) and path.endswith('.csv'):
        csv_files = [path]
    else:
        raise ValueError(f"Path must be a CSV file or a directory containing CSVs: {path}")
    
    if not csv_files:
        raise ValueError(f"No CSV files found in path: {path}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str)
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df['row_index'] = combined_df.index.astype(str)

    records = combined_df.to_dict(orient='records')
    return records
