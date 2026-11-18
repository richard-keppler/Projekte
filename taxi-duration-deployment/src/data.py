from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_NAME = "yellow_tripdata_2025-01.parquet"
FILE_PATH = DATA_DIR / DATA_NAME


def download_data(url, file_path):
    if file_path.exists():
        print(f"Data already exists: {file_path}")
        return file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".part")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tmp_path.write_bytes(response.content)
    tmp_path.rename(file_path)
    print(f"Data saved in: {file_path}")
    return file_path


def load_data(file_path):
    df = pd.read_parquet(file_path)
    return df