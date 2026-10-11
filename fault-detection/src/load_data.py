from pathlib import Path
import pandas as pd

FILE = "detect_dataset.csv"


def load_raw(data_dir=None):
    """Return the raw fault-detection DataFrame exactly as stored in the CSV.
    Nothing is renamed, dropped, converted or cleaned.
    """
    if data_dir is None:  # repo-root/data, works from any working directory
        data_dir = Path(__file__).resolve().parent.parent / "data"
    path = Path(data_dir) / FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download the dataset first.")
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw()
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist(), end="\n\n")
    print(df.head(5).to_string())
