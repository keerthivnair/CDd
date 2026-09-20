def validate(records: list[dict]) -> list[dict]:
    valid_records = []
    seen = set()
    invalid_count = 0
    duplicate_count = 0
    
    for record in records:
        created_at = record.get("created_at")
        text = record.get("original_text")
        
        # Check basic existence
        if pd_isna(created_at) or pd_isna(text) or str(text).strip() == "":
            invalid_count += 1
            continue
            
        # Check duplicates based on text + timestamp
        signature = (str(created_at).strip(), str(text).strip())
        if signature in seen:
            duplicate_count += 1
            continue
            
        seen.add(signature)
        valid_records.append(record)
        
    # Print summary
    print("--- Validation Summary ---")
    print(f"Total records loaded: {len(records)}")
    print(f"Valid records:        {len(valid_records)}")
    print(f"Invalid records:      {invalid_count}")
    print(f"Duplicates:           {duplicate_count}")
    
    if valid_records:
        dates = [r.get("created_at") for r in valid_records]
        print(f"Timestamp range:      {min(dates)} to {max(dates)}")
    
    print("--------------------------")
        
    return valid_records

def pd_isna(val):
    import pandas as pd
    return pd.isna(val) or val is None or str(val).lower() == 'nan'
