import os
import pandas as pd
import numpy as np
from faker import Faker

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "synthetic_crime_data.csv")

# Faker instance
fake = Faker()

# Crime types
crime_types = ["Theft", "Assault", "Burglary", "Fraud", "Robbery"]

# Generate synthetic dataset
rows = []
for _ in range(1000):
    dt = fake.date_time_between(start_date="-3y", end_date="now")
    crime = np.random.choice(crime_types)
    lat = np.random.uniform(12.8, 13.1)
    lon = np.random.uniform(77.4, 77.7)
    area = fake.city()

    rows.append({
        "datetime": dt.isoformat(),  # save as ISO string
        "crime_type": crime,
        "latitude": lat,
        "longitude": lon,
        "area": area
    })

df = pd.DataFrame(rows)

# Save CSV
df.to_csv(DATA_FILE, index=False)
print(f"✅ synthetic_crime_data.csv generated with {len(df)} rows at {DATA_FILE}")
