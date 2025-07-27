import json
import pandas as pd

def dosyaOkuma(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    df = pd.json_normalize(raw_data, record_path=['messages'], meta=['conversation_id'])
    print(df.head())
    return df