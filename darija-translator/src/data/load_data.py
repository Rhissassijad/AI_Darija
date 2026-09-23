from pathlib import Path

import pandas as pd


def load_parallel_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"en", "darija"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {expected_cols}")

    df = df[["en", "darija"]].dropna()
    df["en"] = df["en"].astype(str).str.strip()
    df["darija"] = df["darija"].astype(str).str.strip()
    return df[df["en"].str.len() > 0]
