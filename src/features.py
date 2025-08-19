
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .config import RANDOM_STATE, N_CLUSTERS, HIGH_RISK_TOP_PERCENT

def engineer_classifier_features(df: pd.DataFrame):
    X = df.copy()

    # Simple encodings
    for col in ["crime_type", "area", "month_name"]:
        if col in X.columns:
            X[f"{col}_id"] = X[col].astype("category").cat.codes

    # Build a baseline 'high risk' label by (area,hour) frequency
    stats = (
        X.groupby(["area","hour"], dropna=False)
         .size()
         .rename("count")
         .reset_index()
    )
    if len(stats) == 0:
        # not enough data, return empty
        return X[[]], pd.Series([], dtype=int)

    thr = np.percentile(stats["count"], 100 - HIGH_RISK_TOP_PERCENT) if stats["count"].nunique() > 1 else stats["count"].max()
    stats["is_high_risk"] = (stats["count"] >= thr).astype(int)

    X = X.merge(stats[["area","hour","is_high_risk"]], on=["area","hour"], how="left")
    y = X["is_high_risk"].fillna(0).astype(int)

    # numeric features
    num_cols = ["hour","dayofweek","month","year","crime_type_id","area_id"]
    num_cols = [c for c in num_cols if c in X.columns]
    X_num = X[num_cols].fillna(-1)

    return X_num, y

def kmeans_hotspots(df: pd.DataFrame, k: int = None):
    if not {"latitude","longitude"}.issubset(df.columns):
        return pd.DataFrame(), {"message":"No coordinates"}
    k = k or N_CLUSTERS
    coords = df[["latitude","longitude"]].dropna()
    if coords.empty:
        return pd.DataFrame(), {"message":"No valid coords"}
    if len(coords) < k:
        k = max(1, len(coords))
    if k <= 1:
        labels = np.zeros(len(coords), dtype=int)
        centers = coords.mean().to_frame().T.rename(columns={0:"latitude",1:"longitude"})
        centers.columns = ["latitude","longitude"]
    else:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(coords)
        centers = pd.DataFrame(km.cluster_centers_, columns=["latitude","longitude"])
    out = coords.copy()
    out["cluster"] = labels
    return out, {"centers": centers, "k": k}
