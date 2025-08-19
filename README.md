
# Crime Pattern Analytics & Forecasting (Ready-to-Run)

## Setup
```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Generate data
```bash
py generate_data.py
```

## Train & Forecast
```bash
py -m src.train
py -m src.forecast
```

## Run Dashboard
```bash
streamlit run src/app.py
```
