# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Inference handler for Customer Propensity Scoring model.
Compatible with: local, SageMaker Batch Transform, Clean Rooms ML.
"""

import os, json, logging, io
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
_model = None
_feature_cols = None

CLEANROOMS_COLUMNS = [
    "ad_campaign_id", "impressions", "clicks", "time_spent_seconds",
    "device_type", "event_date", "product_category", "purchase_amount",
    "purchase_count", "site_visits", "days_since_last_purchase",
    "last_purchase_date", "converted"
]


def load_model():
    global _model, _feature_cols
    if _model is not None:
        return _model, _feature_cols

    search_dirs = [MODEL_DIR, "/opt/ml/model", "/opt/ml/input/data/model"]
    model_path = features_path = None

    for d in search_dirs:
        if os.path.exists(d):
            candidate = os.path.join(d, "model.joblib")
            if os.path.exists(candidate):
                model_path = candidate
                features_path = os.path.join(d, "feature_columns.json")
                break

    if model_path is None:
        for root, dirs, files in os.walk("/opt/ml"):
            if "model.joblib" in files:
                model_path = os.path.join(root, "model.joblib")
                features_path = os.path.join(root, "feature_columns.json")
                break

    if model_path is None:
        raise FileNotFoundError(f"model.joblib not found in any of: {search_dirs}")

    _model = joblib.load(model_path)
    if features_path and os.path.exists(features_path):
        with open(features_path, "r") as f:
            _feature_cols = json.load(f)
    else:
        _feature_cols = [
            "impressions", "clicks", "time_spent_seconds", "click_through_rate",
            "time_per_click", "purchase_amount", "purchase_count", "site_visits",
            "days_since_last_purchase",
        ]
    logger.info(f"Model loaded. Features: {_feature_cols}")
    return _model, _feature_cols


MAX_INPUT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit

EXPECTED_COLUMNS = {
    "user_id", "ad_campaign_id", "impressions", "clicks", "time_spent_seconds",
    "device_type", "event_date", "product_category", "purchase_amount",
    "purchase_count", "site_visits", "days_since_last_purchase",
    "last_purchase_date", "converted",
}


def predict(input_data, content_type="text/csv"):
    model, feature_cols = load_model()

    # Input size validation
    if len(input_data.encode("utf-8")) > MAX_INPUT_SIZE_BYTES:
        raise ValueError(f"Input exceeds maximum allowed size of {MAX_INPUT_SIZE_BYTES} bytes")

    # Parse input with error handling
    try:
        if content_type == "application/json":
            df = pd.read_json(io.StringIO(input_data))
        else:
            df = pd.read_csv(io.StringIO(input_data))
            if df.columns[0] not in ["user_id", "ad_campaign_id", "product_category",
                                      "impressions", "clicks"] and len(df.columns) == len(CLEANROOMS_COLUMNS):
                df = pd.read_csv(io.StringIO(input_data), header=None, names=CLEANROOMS_COLUMNS)
    except Exception as e:
        raise ValueError(f"Failed to parse input data ({content_type}): {e}")

    if df.empty:
        raise ValueError("Input data is empty")

    # Schema validation: check that columns are a subset of expected
    unknown_cols = set(df.columns) - EXPECTED_COLUMNS - set(feature_cols)
    if unknown_cols:
        logger.warning(f"Unexpected columns in input (ignored): {unknown_cols}")

    user_ids = df["user_id"] if "user_id" in df.columns else None

    if "click_through_rate" not in df.columns and "impressions" in df.columns:
        df["click_through_rate"] = df["clicks"] / df["impressions"].clip(lower=1)
    if "time_per_click" not in df.columns and "time_spent_seconds" in df.columns:
        df["time_per_click"] = df["time_spent_seconds"] / df["clicks"].clip(lower=1)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    result = pd.DataFrame({
        "propensity_score": np.round(probabilities, 4),
        "predicted_converter": predictions.astype(int),
    })
    if user_ids is not None:
        result.insert(0, "user_id", user_ids.values)

    logger.info(f"Output shape: {result.shape}")
    return result.to_csv(index=False)
