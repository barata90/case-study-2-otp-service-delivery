# Plotly Dash — Airline OTP & Service Delivery Dashboard

This Dash app is built for the Kaggle **Airline Passenger Satisfaction** dataset and mirrors the notebook outputs:
- KPI cards: OTP (<=15), severe-delay rate (>=60), avg delay minutes, satisfaction rate
- Operational charts: arrival-delay histogram, OTP/severe-delay by Class
- Service delivery: average ratings and satisfaction gaps
- Model drivers: permutation importance (raw-feature level)
- Severe-delay risk: threshold slider + confusion matrix + ROC curve (leakage-aware Variant B)

## Files expected
- `train.csv`
- `test.csv`

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

Open:
- http://127.0.0.1:8050

## Optional configuration
Run with a custom dataset folder:
```bash
DATA_DIR=/path/to/folder python app.py
```

Run on a custom port:
```bash
PORT=8060 python app.py
```
