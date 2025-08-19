
import os
import pandas as pd
import numpy as np
from .config import COLUMN_MAP, DATA_DIR, DATA_FILE

def read_data(path: str = None) -> pd.DataFrame:
    """Read CSV and normalize expected column names per COLUMN_MAP."""
    path = path or os.path.join(DATA_DIR, DATA_FILE)
    df = pd.read_csv(path)
    # Normalize column names
    rename_map = {}
    for std, col in COLUMN_MAP.items():
        if col in df.columns and std != col:
            rename_map[col] = std
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "crime_type" in df.columns:
        df["crime_type"] = df["crime_type"].astype(str).str.strip().str.title()
    if "area" in df.columns:
        df["area"] = df["area"].astype(str).str.strip().str.title()
    # validate lat/lon ranges if present
    if {"latitude","longitude"}.issubset(df.columns):
        df.loc[(df["latitude"].abs() > 90), "latitude"] = np.nan
        df.loc[(df["longitude"].abs() > 180), "longitude"] = np.nan
    return df.drop_duplicates()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month_name"] = dt.dt.month_name()
    return df

def summarize(df: pd.DataFrame):
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "min_date": str(df["datetime"].min()) if "datetime" in df.columns else "N/A",
        "max_date": str(df["datetime"].max()) if "datetime" in df.columns else "N/A",
        "unique_types": int(df["crime_type"].nunique()) if "crime_type" in df.columns else 0,
        "unique_areas": int(df["area"].nunique()) if "area" in df.columns else 0,
        "has_geo": int({"latitude","longitude"}.issubset(df.columns)),
    }
