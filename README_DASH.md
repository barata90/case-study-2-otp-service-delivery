# Case Study 2 — Plotly Dash Dashboard (Operational Performance & Service Delivery)

This Dash app uses the Kaggle **Airline Passenger Satisfaction** dataset and mirrors the notebook outputs in an interactive dashboard:
- KPI cards: OTP (<=15), severe-delay rate (>=60), avg delay minutes, satisfaction rate
- Operational charts: arrival-delay histogram, OTP/severe-delay by Class
- Service delivery: average ratings and satisfaction gaps
- Model drivers: permutation importance (raw-feature level)
- Severe-delay risk: threshold slider + confusion matrix + ROC curve (leakage-aware Variant B)

## Folder layout (recommended)
Place these files in the same folder:

- `app.py`
- `requirements.txt`
- `README_DASH.md`
- `train.csv`
- `test.csv`

> The app automatically loads `train.csv` and `test.csv` from the **same directory as `app.py`**.  
> If the CSVs are in a different folder, set `DATA_DIR` as shown below.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

Open:
- http://127.0.0.1:8050

## Optional configuration
Use a custom dataset folder:
```bash
DATA_DIR=/path/to/csv/folder python app.py
```

Change port:
```bash
PORT=8060 python app.py
```

Enable Dash debug (off by default):
```bash
DASH_DEBUG=1 python app.py
```
