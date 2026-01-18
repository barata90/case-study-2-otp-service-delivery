#!/usr/bin/env python3
"""
Plotly Dash — Airline Operational Performance & Service Delivery Dashboard
Dataset: Kaggle Airline Passenger Satisfaction (train.csv / test.csv)

Run:
  pip install -r requirements.txt
  python app.py

Environment variables (optional):
  DATA_DIR: folder containing train.csv and test.csv (default: current working directory)
  PORT: server port (default: 8050)
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42


def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.getenv("DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {os.path.abspath(data_dir)}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {os.path.abspath(data_dir)}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # Drop index-like column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Median imputation for arrival delay (robust to long tail)
    if "Arrival Delay in Minutes" in df.columns:
        med = df["Arrival Delay in Minutes"].median()
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(med)

    # KPI labels
    if "Arrival Delay in Minutes" in df.columns:
        df["otp_arrival_15"] = (df["Arrival Delay in Minutes"] <= 15).astype(int)
        df["severe_delay_60"] = (df["Arrival Delay in Minutes"] >= 60).astype(int)
    else:
        df["otp_arrival_15"] = np.nan
        df["severe_delay_60"] = np.nan

    # Rating columns used for service-quality index
    rating_cols = [
        c for c in df.columns
        if c not in ["id", "satisfaction", "otp_arrival_15", "severe_delay_60"]
        and df[c].dtype != "object"
        and c not in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    ]
    if rating_cols:
        df["service_quality_index"] = df[rating_cols].mean(axis=1)
    else:
        df["service_quality_index"] = np.nan

    return df, rating_cols


def add_delay_anomaly(train_c: pd.DataFrame, test_c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    iso_cols = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
    iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    iso.fit(train_c[iso_cols])
    train_c = train_c.copy()
    test_c = test_c.copy()
    train_c["delay_anomaly"] = (iso.predict(train_c[iso_cols]) == -1).astype(int)
    test_c["delay_anomaly"] = (iso.predict(test_c[iso_cols]) == -1).astype(int)
    return train_c, test_c


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "records": int(len(df)),
        "otp_arrival_15_rate": float(df["otp_arrival_15"].mean()) if len(df) else np.nan,
        "avg_departure_delay_min": float(df["Departure Delay in Minutes"].mean()) if len(df) else np.nan,
        "avg_arrival_delay_min": float(df["Arrival Delay in Minutes"].mean()) if len(df) else np.nan,
        "severe_delay_60_rate": float(df["severe_delay_60"].mean()) if len(df) else np.nan,
        "satisfaction_rate": float((df["satisfaction"] == "satisfied").mean()) if ("satisfaction" in df.columns and len(df)) else np.nan,
    }


def build_preprocess(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
    )
    return preprocess, cat_cols, num_cols


@dataclass
class SatisfactionArtifacts:
    best_name: str
    best_model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    proba: np.ndarray
    roc_points: tuple[np.ndarray, np.ndarray, np.ndarray]
    cm: np.ndarray
    metrics: dict
    perm_importance: pd.DataFrame


@dataclass
class SevereDelayArtifacts:
    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    proba: np.ndarray
    roc_points: tuple[np.ndarray, np.ndarray, np.ndarray]
    base_metrics: dict



def compute_perm_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Lazy permutation importance (raw-feature level).
    Computed on-demand to avoid slow app startup.

    Controls (optional env vars):
      PI_SAMPLE   default 5000
      PI_REPEATS  default 3
      PI_NJOBS    default 1  (set -1 to use all cores)
    """
    sample_n = int(os.getenv("PI_SAMPLE", "5000"))
    repeats = int(os.getenv("PI_REPEATS", "3"))
    n_jobs = int(os.getenv("PI_NJOBS", "1"))

    sample_n = min(sample_n, len(X_test))
    X_pi = X_test.sample(sample_n, random_state=RANDOM_STATE)
    y_pi = y_test.loc[X_pi.index]

    pi = permutation_importance(
        model, X_pi, y_pi,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )

    perm = pd.DataFrame({
        "feature": X_pi.columns.to_numpy(),
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return perm

def train_satisfaction_model(train_c: pd.DataFrame, test_c: pd.DataFrame) -> SatisfactionArtifacts:
    target_col = "satisfaction"
    has_label_test = target_col in test_c.columns

    y_train = (train_c[target_col] == "satisfied").astype(int)
    X_train = train_c.drop(columns=[target_col])

    if has_label_test:
        y_test = (test_c[target_col] == "satisfied").astype(int)
        X_test = test_c.drop(columns=[target_col])
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )

    # Column alignment safety
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    preprocess, _, _ = build_preprocess(X_train)

    rf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )),
        ]
    )

    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    fpr, tpr, thr = roc_curve(y_test, proba)

    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "support_0": int((y_test == 0).sum()),
        "support_1": int((y_test == 1).sum()),
    }

    # Permutation importance is computed lazily (on-demand) in the dashboard tab.
    perm = pd.DataFrame(columns=['feature','importance_mean','importance_std'])

    return SatisfactionArtifacts(
        best_name="Random Forest",
        best_model=rf,
        X_test=X_test,
        y_test=y_test,
        proba=proba,
        roc_points=(fpr, tpr, thr),
        cm=cm,
        metrics=metrics,
        perm_importance=perm,
    )


def train_severe_delay_model(train_c: pd.DataFrame) -> SevereDelayArtifacts:
    # Target: severe delay (>= 60 minutes)
    y = train_c["severe_delay_60"].astype(int)

    base_drop = ["satisfaction", "satisfaction_bin", "severe_delay_60", "otp_arrival_15", "delay_anomaly"]
    available_cols = [c for c in train_c.columns if c not in base_drop]

    # Variant B: include departure delay, exclude arrival delay (leakage-aware)
    X = train_c[available_cols].drop(columns=["Arrival Delay in Minutes"], errors="ignore")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocess, _, _ = build_preprocess(X_tr)

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    fpr, tpr, thr = roc_curve(y_te, proba)

    # Baseline metrics at threshold 0.50
    pred = (proba >= 0.5).astype(int)
    base_metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "support_0": int((y_te == 0).sum()),
        "support_1": int((y_te == 1).sum()),
    }

    return SevereDelayArtifacts(
        model=model,
        X_test=X_te,
        y_test=y_te,
        proba=proba,
        roc_points=(fpr, tpr, thr),
        base_metrics=base_metrics,
    )


# ----------------------------
# Load + prepare data once
# ----------------------------
train, test = load_csvs()
train_c, rating_cols = clean_frame(train)
test_c, _ = clean_frame(test)

# Ensure engineered columns are consistent across splits
train_c, test_c = add_delay_anomaly(train_c, test_c)

# Precompute service aggregates (train-level)
avg_ratings = train_c[rating_cols].mean().sort_values() if rating_cols else pd.Series(dtype=float)

train_c["satisfaction_bin"] = (train_c["satisfaction"] == "satisfied").astype(int) if "satisfaction" in train_c.columns else np.nan
gap = (
    train_c.groupby("satisfaction_bin")[rating_cols].mean()
    .T.rename(columns={0: "neutral_or_dissatisfied", 1: "satisfied"})
) if rating_cols else pd.DataFrame()
if not gap.empty:
    gap["satisfaction_gap"] = gap["satisfied"] - gap["neutral_or_dissatisfied"]
    gap = gap.sort_values("satisfaction_gap")

# Train models (startup artifacts)
sat_art = train_satisfaction_model(train_c, test_c)
delay_art = train_severe_delay_model(train_c)


# ----------------------------
# Dash app
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Case Study 2 — Operational Performance & Service Delivery"


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.1f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


def kpi_card(title: str, value: str, subtitle: str = ""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.Div(value, className="h3"),
            html.Div(subtitle, className="small text-muted"),
        ]),
        className="shadow-sm",
    )


# Filter controls
class_options = [{"label": str(v), "value": v} for v in sorted(train_c["Class"].dropna().unique())]
travel_options = [{"label": str(v), "value": v} for v in sorted(train_c["Type of Travel"].dropna().unique())]
cust_options = [{"label": str(v), "value": v} for v in sorted(train_c["Customer Type"].dropna().unique())]

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("Case Study 2 — Operational Performance & Service Delivery"), width=9),
        dbc.Col(html.Div("Dataset: Airline Passenger Satisfaction (Kaggle)", className="text-muted"), width=3),
    ], className="mt-3 mb-2"),

    dbc.Card(className="shadow-sm mb-3", body=True, children=[
        dbc.Row([
            dbc.Col([
                html.Div("Filters", className="text-muted mb-2"),
                html.Label("Class"),
                dcc.Dropdown(
                    id="f-class", options=class_options, value=[],
                    multi=True, placeholder="All"
                ),
            ], md=4),
            dbc.Col([
                html.Label("Type of Travel"),
                dcc.Dropdown(
                    id="f-travel", options=travel_options, value=[],
                    multi=True, placeholder="All"
                ),
            ], md=4),
            dbc.Col([
                html.Label("Customer Type"),
                dcc.Dropdown(
                    id="f-customer", options=cust_options, value=[],
                    multi=True, placeholder="All"
                ),
            ], md=4),
        ]),
        html.Hr(),
        html.Div("Filtered views update KPI cards and exploratory charts. Model diagnostics remain overall (holdout-based).",
                 className="small text-muted"),
    ]),

    dcc.Tabs(id="tabs", value="tab-overview", children=[
        dcc.Tab(label="Overview", value="tab-overview"),
        dcc.Tab(label="Service Delivery", value="tab-service"),
        dcc.Tab(label="Model Drivers", value="tab-drivers"),
        dcc.Tab(label="Severe Delay Risk", value="tab-risk"),
    ]),
    html.Div(id="tab-content", className="mt-3"),
])


def filter_df(df: pd.DataFrame, classes, travels, customers) -> pd.DataFrame:
    out = df
    if classes:
        out = out[out["Class"].isin(classes)]
    if travels:
        out = out[out["Type of Travel"].isin(travels)]
    if customers:
        out = out[out["Customer Type"].isin(customers)]
    return out


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("f-class", "value"),
    Input("f-travel", "value"),
    Input("f-customer", "value"),
)
def render_tab(tab, f_class, f_travel, f_customer):
    dff = filter_df(train_c, f_class, f_travel, f_customer)

    kpis = kpi_summary(dff)

    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Records", f"{kpis['records']:,}", "Filtered subset"), md=2),
        dbc.Col(kpi_card("OTP (Arrival ≤ 15)", fmt_pct(kpis["otp_arrival_15_rate"]), "15-minute rule"), md=2),
        dbc.Col(kpi_card("Severe delay (≥ 60)", fmt_pct(kpis["severe_delay_60_rate"]), "Tail risk indicator"), md=2),
        dbc.Col(kpi_card("Avg dep delay (min)", fmt_num(kpis["avg_departure_delay_min"]), ""), md=2),
        dbc.Col(kpi_card("Avg arr delay (min)", fmt_num(kpis["avg_arrival_delay_min"]), ""), md=2),
        dbc.Col(kpi_card("Satisfaction rate", fmt_pct(kpis["satisfaction_rate"]), "Train label"), md=2),
    ], className="g-2 mb-3")

    if tab == "tab-overview":
        clip_max = 200
        arr = dff["Arrival Delay in Minutes"].clip(upper=clip_max)
        fig_hist = px.histogram(
            arr, nbins=40,
            title=f"Arrival Delay Distribution (clipped at {clip_max} minutes)"
        )
        fig_hist.update_layout(xaxis_title="Arrival Delay (minutes)", yaxis_title="Count")

        otp_by_class = dff.groupby("Class")["otp_arrival_15"].mean().sort_values()
        fig_otp = px.bar(
            otp_by_class,
            title="OTP (Arrival ≤ 15 min) by Class",
        )
        fig_otp.update_layout(xaxis_title="Class", yaxis_title="OTP rate")

        severe_by_class = dff.groupby("Class")["severe_delay_60"].mean().sort_values()
        fig_sev = px.bar(
            severe_by_class,
            title="Severe delay rate (Arrival ≥ 60 min) by Class",
        )
        fig_sev.update_layout(xaxis_title="Class", yaxis_title="Severe-delay rate")

        return dbc.Container(fluid=True, children=[
            kpi_row,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hist), md=6),
                dbc.Col(dcc.Graph(figure=fig_otp), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sev), md=6),
                dbc.Col(html.Div([
                    html.H5("Holdout model performance (Satisfaction)", className="mt-2"),
                    html.Ul([
                        html.Li(f"Model: {sat_art.best_name}"),
                        html.Li(f"Accuracy: {sat_art.metrics['accuracy']:.3f}"),
                        html.Li(f"ROC-AUC: {sat_art.metrics['roc_auc']:.3f}"),
                        html.Li("Interpretation: service ratings + travel context dominate satisfaction prediction in this dataset."),
                    ], className="small"),
                ], className="p-3"), md=6),
            ], className="g-3 mt-1"),
        ])

    if tab == "tab-service":
        if avg_ratings.empty or gap.empty:
            return dbc.Container(fluid=True, children=[kpi_row, html.Div("Service rating columns not detected.", className="text-muted")])

        fig_avg = px.bar(
            avg_ratings.sort_values(),
            orientation="h",
            title="Average Service Ratings (Train)",
        )
        fig_avg.update_layout(xaxis_title="Average rating", yaxis_title="")

        gap_sorted = gap.sort_values("satisfaction_gap")
        fig_gap = px.bar(
            gap_sorted["satisfaction_gap"],
            orientation="h",
            title="Service Rating Gaps (Satisfied minus Neutral/Dissatisfied)",
        )
        fig_gap.update_layout(xaxis_title="Average rating difference", yaxis_title="")

        return dbc.Container(fluid=True, children=[
            kpi_row,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_avg), md=6),
                dbc.Col(dcc.Graph(figure=fig_gap), md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Top differentiators (gap view)", className="mt-2"),
                    dbc.Table.from_dataframe(
                        gap.sort_values("satisfaction_gap").tail(10).reset_index().rename(columns={"index": "dimension"}),
                        striped=True, bordered=False, hover=True, size="sm"
                    ),
                ], className="p-2"), md=12),
            ])
        ])

    if tab == "tab-drivers":
        # Lazy compute permutation importance to keep startup fast
        if sat_art.perm_importance is None or getattr(sat_art.perm_importance, "empty", True):
            sat_art.perm_importance = compute_perm_importance(sat_art.best_model, sat_art.X_test, sat_art.y_test)

        top_n = 20
        imp = sat_art.perm_importance.head(top_n).iloc[::-1]
        fig_imp = px.bar(
            imp,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title=f"Permutation Importance (ROC-AUC drop) — Top {top_n} features",
        )
        fig_imp.update_layout(xaxis_title="Mean importance (Δ ROC-AUC)", yaxis_title="")

        return dbc.Container(fluid=True, children=[
            kpi_row,
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(figure=fig_imp), type="default"), md=7),
                dbc.Col(html.Div([
                    html.H5("Interpretation notes", className="mt-2"),
                    html.Ul([
                        html.Li("Permutation importance measures performance drop when a feature is shuffled."),
                        html.Li("Higher values indicate stronger contribution to discrimination."),
                        html.Li("Identifiers (e.g., id) can show spurious importance and are typically removed in production pipelines."),
                    ], className="small"),
                ], className="p-3"), md=5),
            ], className="g-3"),
        ])

    if tab == "tab-risk":
        # ROC curve for severe-delay model
        fpr, tpr, _ = delay_art.roc_points
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        fig_roc.update_layout(
            title=f"Severe-delay ROC (Variant B) — AUC {delay_art.base_metrics['roc_auc']:.3f}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420,
        )

        return dbc.Container(fluid=True, children=[
            kpi_row,
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Threshold control", className="mt-2"),
                    html.Div("Adjust threshold to trade off false alarms vs missed severe delays.", className="small text-muted mb-2"),
                    dcc.Slider(
                        id="risk-threshold",
                        min=0.05, max=0.95, step=0.01, value=0.50,
                        marks={0.1:"0.10", 0.3:"0.30", 0.5:"0.50", 0.7:"0.70", 0.9:"0.90"},
                    ),
                    html.Div(id="risk-metrics", className="mt-3"),
                ], className="p-2"), md=5),
                dbc.Col(dcc.Graph(id="risk-cm"), md=7),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=12),
            ], className="g-3 mt-1"),
        ])

    return dbc.Container(fluid=True, children=[kpi_row, html.Div("Tab not available.", className="text-muted")])


@app.callback(
    Output("risk-cm", "figure"),
    Output("risk-metrics", "children"),
    Input("risk-threshold", "value"),
)
def update_risk_threshold(threshold: float):
    y = delay_art.y_test.values
    proba = delay_art.proba
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — Severe delay (threshold={threshold:.2f})",
        xaxis_title="Prediction",
        yaxis_title="Actual",
        height=420,
    )
    annotations = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=j, y=i, text=str(val),
                showarrow=False,
                font=dict(color="white"),
            ))
    fig.update_layout(annotations=annotations)

    metrics = html.Ul([
        html.Li(f"Accuracy: {acc:.3f}"),
        html.Li(f"Precision (class 1): {prec:.3f}"),
        html.Li(f"Recall (class 1): {rec:.3f}"),
        html.Li(f"F1 (class 1): {f1:.3f}"),
        html.Li(f"TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}"),
    ], className="small")

    return fig, metrics


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "0") == "1"
    app.run_server(host="0.0.0.0", port=port, debug=debug)
