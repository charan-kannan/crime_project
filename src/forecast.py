
import os, warnings, joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import read_data, basic_clean, add_time_features
from .config import MODELS_DIR, REPORTS_DIR, TOP_N_TYPES, RESAMPLE_RULE

warnings.filterwarnings("ignore")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def fit_sarimax(series: pd.Series):
    model = SARIMAX(series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_counts(df: pd.DataFrame, steps: int = 6):
    # overall
    ts_all = (
        df.set_index("datetime").sort_index().resample(RESAMPLE_RULE)["crime_type"].count()
    )
    if ts_all.sum() == 0 or len(ts_all) < 3:
        raise ValueError("Time series too short for forecasting.")
    model_all = fit_sarimax(ts_all)
    fc_all = model_all.get_forecast(steps=steps)
    overall = pd.DataFrame({
        "date": fc_all.predicted_mean.index,
        "forecast_count": fc_all.predicted_mean.values
    })
    joblib.dump(model_all, os.path.join(MODELS_DIR, "time_series_model.pkl"))
    overall.to_csv(os.path.join(REPORTS_DIR, "forecast_overall.csv"), index=False)

    # per-type
    top_types = df["crime_type"].value_counts().head(TOP_N_TYPES).index.tolist()
    per_type = []
    for t in top_types:
        dft = df[df["crime_type"] == t]
        ts = dft.set_index("datetime").sort_index().resample(RESAMPLE_RULE)["crime_type"].count()
        if ts.sum() == 0 or len(ts) < 3:
            continue
        m = fit_sarimax(ts)
        fc = m.get_forecast(steps=steps)
        tmp = pd.DataFrame({
            "crime_type": t,
            "date": fc.predicted_mean.index,
            "forecast_count": fc.predicted_mean.values
        })
        per_type.append(tmp)
    if per_type:
        pd.concat(per_type, ignore_index=True).to_csv(
            os.path.join(REPORTS_DIR, "forecast_by_type.csv"), index=False
        )

    return overall

def main():
    df = read_data()
    df = basic_clean(df)
    df = add_time_features(df)
    _ = forecast_counts(df, steps=6)
    print("Saved reports/forecast_overall.csv (and per-type if available).")

if __name__ == "__main__":
    main()
