
import os

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# Data
DATA_FILE = "synthetic_crime_data.csv"
COLUMN_MAP = {
    "incident_id": "incident_id",
    "datetime": "datetime",
    "crime_type": "crime_type",
    "latitude": "latitude",
    "longitude": "longitude",
    "area": "area",
}

# Modeling / features
RANDOM_STATE = 42
N_CLUSTERS = 5
HIGH_RISK_TOP_PERCENT = 20  # top 20% of (area,hour) counts labeled high-risk for baseline
RESAMPLE_RULE = "M"
TOP_N_TYPES = 5
