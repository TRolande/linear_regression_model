"""
FastAPI service for the African Inflation Risk Prediction model (Task 2).

Loads the artifacts produced by Task 1's notebook:
    - best_model.joblib      (winning regressor, chosen by lowest test MSE)
    - scaler.joblib          (StandardScaler fit on training data)
    - banking_encoder.joblib (LabelEncoder for the banking_crisis column)
    - feature_names.json     (exact column order the model expects)
    - model_metadata.json    (which model won, and why)

Endpoints:
    GET  /              - basic liveness message
    GET  /health        - health check
    POST /predict       - main prediction endpoint
    POST /retrain       - upload a new CSV to retrain the model in place
"""

import json
import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "banking_encoder.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

COUNTRIES = [
    "Algeria", "Angola", "Central African Republic", "Egypt", "Ivory Coast",
    "Kenya", "Mauritius", "Morocco", "Nigeria", "South Africa", "Tunisia",
    "Zambia", "Zimbabwe",
]

app = FastAPI(
    title="InflaTrack Africa API",
    description=(
        "Predicts annual CPI inflation risk for African economies from historical "
        "financial-crisis indicators (systemic, currency, sovereign-debt, and banking crises)."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# Following the pattern from FastAPI's own CORS documentation: build an explicit
# `origins` list and pass it to CORSMiddleware, rather than reaching for the "*"
# wildcard shortcut.
#
# Reasoning for each setting:
# - allow_origins lists real origins a browser-based client would use to reach
#   this API during local development (a Flutter *web* build, or a browser
#   hitting Swagger UI's "Try it out" from a different port). A native Flutter
#   *mobile* app never sends a browser Origin header at all, so CORS doesn't
#   apply to it either way — this list only matters if a web frontend is added.
#   Replace/extend with your actual deployed frontend domain once you have one.
# - allow_methods is restricted to GET and POST, since this API only ever
#   needs to read (GET /health) or submit data (POST /predict, POST /retrain) —
#   never PUT/PATCH/DELETE. Narrowing the verb list reduces attack surface
#   versus a blanket "*".
# - allow_headers is restricted to Content-Type (required for JSON POST bodies)
#   and Authorization (reserved for future auth; unused today but harmless to
#   allow ahead of time).
# - allow_credentials is False: this API doesn't use cookies or session auth,
#   so there's nothing to protect by turning it on. Per FastAPI's docs, note
#   that allow_origins, allow_methods, and allow_headers can NOT be wildcarded
#   if allow_credentials were ever set to True — another reason to keep all
#   three explicit now, so enabling credentials later doesn't silently break.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artifacts at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
banking_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

with open(FEATURES_PATH) as f:
    FEATURE_NAMES = json.load(f)

with open(METADATA_PATH) as f:
    METADATA = json.load(f)


# ---------------------------------------------------------------------------
# Pydantic request schema
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    """
    Input schema for a single inflation-risk prediction.
    Every field enforces both a data type and a realistic range, matching the
    bounds actually observed in the training data (see Task 1 notebook, Section 3).
    """

    year: int = Field(
        ..., ge=1860, le=2014,
        description="Calendar year of observation (1860-2014).",
        json_schema_extra={"example": 2010},
    )
    systemic_crisis: Literal[0, 1] = Field(
        ..., description="1 if a systemic financial crisis occurred that year, else 0.",
        json_schema_extra={"example": 0},
    )
    exch_usd: float = Field(
        ..., ge=0.0, le=1000.0,
        description="Local currency exchange rate against the USD.",
        json_schema_extra={"example": 5.2},
    )
    domestic_debt_in_default: Literal[0, 1] = Field(
        ..., description="1 if the country defaulted on domestic debt that year, else 0.",
        json_schema_extra={"example": 0},
    )
    sovereign_external_debt_default: Literal[0, 1] = Field(
        ..., description="1 if the country defaulted on external sovereign debt that year, else 0.",
        json_schema_extra={"example": 0},
    )
    gdp_weighted_default: float = Field(
        ..., ge=0.0, le=1.0,
        description="GDP-weighted default measure across the region (0-1 scale).",
        json_schema_extra={"example": 0.0},
    )
    independence: Literal[0, 1] = Field(
        ..., description="1 if the country was independent that year, else 0.",
        json_schema_extra={"example": 1},
    )
    currency_crises: Literal[0, 1, 2] = Field(
        ..., description="Currency crisis severity flag (0 = none, 1 = crisis, 2 = severe).",
        json_schema_extra={"example": 0},
    )
    inflation_crises: Literal[0, 1] = Field(
        ..., description="1 if inflation crossed crisis thresholds that year, else 0.",
        json_schema_extra={"example": 0},
    )
    banking_crisis: Literal["crisis", "no_crisis"] = Field(
        ..., description="Whether a banking crisis occurred that year.",
        json_schema_extra={"example": "no_crisis"},
    )
    country: Literal[
        "Algeria", "Angola", "Central African Republic", "Egypt", "Ivory Coast",
        "Kenya", "Mauritius", "Morocco", "Nigeria", "South Africa", "Tunisia",
        "Zambia", "Zimbabwe",
    ] = Field(
        ..., description="Country of observation.",
        json_schema_extra={"example": "Kenya"},
    )


class PredictionOutput(BaseModel):
    predicted_log_inflation: float
    predicted_inflation_annual_cpi_percent: float
    model_used: str
    risk_level: str
    explanation: str


class RetrainOutput(BaseModel):
    message: str
    rows_used: int
    new_best_model: str
    metrics: dict


def generate_explanation(payload: PredictionInput, pred_percent: float) -> tuple[str, str]:
    if pred_percent < 5.0:
        risk_level = "Low Risk"
    elif pred_percent < 15.0:
        risk_level = "Moderate Risk"
    elif pred_percent < 30.0:
        risk_level = "High Risk"
    else:
        risk_level = "Severe Crisis Risk"

    active_crises = []
    if payload.systemic_crisis == 1:
        active_crises.append("systemic financial crisis")
    if payload.domestic_debt_in_default == 1:
        active_crises.append("domestic debt default")
    if payload.sovereign_external_debt_default == 1:
        active_crises.append("sovereign external debt default")
    if payload.currency_crises > 0:
        active_crises.append("currency crisis")
    if payload.inflation_crises == 1:
        active_crises.append("inflation crisis flag")
    if payload.banking_crisis == "crisis":
        active_crises.append("banking crisis")

    if active_crises:
        crisis_summary = f"Active stress factors ({', '.join(active_crises)}) increase economic instability."
    else:
        crisis_summary = "No active banking, debt default, or currency crises are present, indicating stable economic conditions."

    explanation = (
        f"The predicted annual CPI inflation for {payload.country} in {payload.year} is {pred_percent:.2f}% ({risk_level}). "
        f"Exchange rate is {payload.exch_usd:.1f} USD. {crisis_summary}"
    )

    return risk_level, explanation


# ---------------------------------------------------------------------------
# Helper: build the full model-ready feature row from a validated input
# ---------------------------------------------------------------------------
def build_feature_row(payload: PredictionInput) -> pd.DataFrame:
    banking_crisis_enc = banking_encoder.transform([payload.banking_crisis])[0]

    row = {
        "year": payload.year,
        "systemic_crisis": payload.systemic_crisis,
        "exch_usd": payload.exch_usd,
        "domestic_debt_in_default": payload.domestic_debt_in_default,
        "sovereign_external_debt_default": payload.sovereign_external_debt_default,
        "gdp_weighted_default": payload.gdp_weighted_default,
        "independence": payload.independence,
        "currency_crises": payload.currency_crises,
        "inflation_crises": payload.inflation_crises,
        "banking_crisis_enc": banking_crisis_enc,
    }

    # one-hot country columns, matching training (drop_first=True dropped "Algeria")
    for c in COUNTRIES:
        col = f"country_{c}"
        if col in FEATURE_NAMES:
            row[col] = 1 if payload.country == c else 0

    df_row = pd.DataFrame([row])
    # reindex to guarantee exact column order the model was trained on
    df_row = df_row.reindex(columns=FEATURE_NAMES, fill_value=0)
    return df_row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "InflaTrack Africa API is running. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_in_use": METADATA.get("best_model_name")}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    try:
        feature_row = build_feature_row(payload)
        scaled = scaler.transform(feature_row)
        pred_log = float(model.predict(scaled)[0])
        pred_real = float(np.sign(pred_log) * np.expm1(np.abs(pred_log)))
        risk_level, explanation = generate_explanation(payload, pred_real)
        return PredictionOutput(
            predicted_log_inflation=pred_log,
            predicted_inflation_annual_cpi_percent=pred_real,
            model_used=METADATA.get("best_model_name", "unknown"),
            risk_level=risk_level,
            explanation=explanation,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/retrain", response_model=RetrainOutput)
async def retrain(file: UploadFile = File(...)):
    """
    Accepts a new CSV file (same schema as african_crises.csv) and retrains all
    4 candidate models on it, saving whichever comes out best as the new
    best_model.joblib. This satisfies the "trigger retraining when new data is
    uploaded" requirement — call this endpoint via Swagger UI's file upload
    widget to trigger it manually.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        raw_bytes = await file.read()
        tmp_path = os.path.join(MODEL_DIR, "_incoming.csv")
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        new_df = pd.read_csv(tmp_path)

        # --- same cleaning logic as the Task 1 notebook ---
        d = new_df.copy()
        str_cols = d.select_dtypes(include="object").columns
        for c in str_cols:
            d[c] = d[c].astype(str).str.strip()
        d = d.drop_duplicates()
        d = d.dropna(subset=["inflation_annual_cpi"])
        num_cols = d.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if d[c].isna().sum() > 0:
                d[c] = d[c].fillna(d[c].median())
        for c in str_cols:
            if d[c].isna().sum() > 0:
                d[c] = d[c].fillna(d[c].mode()[0])
        d = d.drop(columns=["case", "cc3"], errors="ignore")

        d["log_inflation"] = np.sign(d["inflation_annual_cpi"]) * np.log1p(
            np.abs(d["inflation_annual_cpi"])
        )

        new_encoder = LabelEncoder()
        d["banking_crisis_enc"] = new_encoder.fit_transform(d["banking_crisis"])
        country_dummies = pd.get_dummies(d["country"], prefix="country", drop_first=True)

        feature_cols_base = [
            "year", "systemic_crisis", "exch_usd", "domestic_debt_in_default",
            "sovereign_external_debt_default", "gdp_weighted_default", "independence",
            "currency_crises", "inflation_crises", "banking_crisis_enc",
        ]
        X_new = pd.concat([d[feature_cols_base], country_dummies], axis=1)
        y_new = d["log_inflation"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )

        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)

        candidates = {
            "Linear Regression (OLS)": LinearRegression(),
            "SGD Regressor (Gradient Descent)": SGDRegressor(
                max_iter=1000, random_state=42
            ),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, max_depth=8, random_state=42
            ),
        }

        scored = {}
        for name, m in candidates.items():
            m.fit(X_train_scaled, y_train)
            pred = m.predict(X_test_scaled)
            scored[name] = {
                "model": m,
                "mse": mean_squared_error(y_test, pred),
                "r2": r2_score(y_test, pred),
            }

        best_name = min(scored, key=lambda k: scored[k]["mse"])
        best_model_obj = scored[best_name]["model"]

        # overwrite the artifacts this API loads on next startup / next call
        joblib.dump(best_model_obj, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        joblib.dump(new_encoder, ENCODER_PATH)
        with open(FEATURES_PATH, "w") as f:
            json.dump(list(X_new.columns), f)

        new_metadata = {
            "best_model_name": best_name,
            "target": "log_inflation (signed log1p of inflation_annual_cpi)",
            "results": {k: {"mse": v["mse"], "r2": v["r2"]} for k, v in scored.items()},
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(new_metadata, f, indent=2)

        # hot-swap the in-memory objects so /predict uses the new model immediately
        global model, scaler, banking_encoder, FEATURE_NAMES, METADATA
        model = best_model_obj
        scaler = new_scaler
        banking_encoder = new_encoder
        FEATURE_NAMES = list(X_new.columns)
        METADATA = new_metadata

        os.remove(tmp_path)

        return RetrainOutput(
            message="Model retrained successfully on uploaded data.",
            rows_used=len(d),
            new_best_model=best_name,
            metrics={k: {"mse": v["mse"], "r2": v["r2"]} for k, v in scored.items()},
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Retraining failed: {exc}") from exc