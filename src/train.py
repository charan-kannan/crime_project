
import os, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from .utils import read_data, basic_clean, add_time_features, summarize
from .features import engineer_classifier_features, kmeans_hotspots
from .config import MODELS_DIR, REPORTS_DIR, N_CLUSTERS

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    df = read_data()
    df = basic_clean(df)
    df = add_time_features(df)
    print("DATA SUMMARY:", summarize(df))

    # Hotspots
    coords, meta = kmeans_hotspots(df, k=N_CLUSTERS)
    if not coords.empty:
        coords.to_csv(os.path.join(REPORTS_DIR, "hotspot_points.csv"), index=False)
        meta["centers"].to_csv(os.path.join(REPORTS_DIR, "hotspot_centers.csv"), index=False)
        print(f"Saved hotspot_points.csv & hotspot_centers.csv (k={meta['k']})")
    else:
        print("No coordinates — skipping KMeans.")

    # Classifier
    X, y = engineer_classifier_features(df)
    if X.empty or y.empty:
        print("Not enough data to train classifier.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=1000))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier.pkl"))
    print("Saved model → models/classifier.pkl")

if __name__ == "__main__":
    main()
