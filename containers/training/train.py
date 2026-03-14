# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Training script for Customer Propensity Scoring model.
Compatible with: local testing, SageMaker Training, and Clean Rooms ML.
"""

import argparse, os, sys, json, glob, traceback, logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLEANROOMS_COLUMNS = [
    "ad_campaign_id", "impressions", "clicks", "time_spent_seconds",
    "device_type", "event_date", "product_category", "purchase_amount",
    "purchase_count", "site_visits", "days_since_last_purchase",
    "last_purchase_date", "converted"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data"))
    parser.add_argument("--train_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN",
                                os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--train_file_format", type=str, default=os.environ.get("FILE_FORMAT", "csv"))
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def load_data(train_dir, file_format):
    logger.info(f"Loading data from {train_dir} (format: {file_format})")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            logger.info(f"  Dir: {root}, subdirs: {dirs}, files: {files}")
    else:
        alternatives = ["/opt/ml/input/data/training", "/opt/ml/input/data/train", "/opt/ml/input/data"]
        for alt in alternatives:
            if os.path.exists(alt):
                train_dir = alt
                break
        else:
            raise FileNotFoundError(f"No training data directory found. Tried: {train_dir}, {alternatives}")

    all_files = []
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            if f.endswith(f".{file_format}") or not os.path.splitext(f)[1]:
                all_files.append(os.path.join(root, f))
    if not all_files:
        all_files = [f for f in glob.glob(os.path.join(train_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    if not all_files:
        raise FileNotFoundError(f"No data files found in {train_dir}")

    logger.info(f"Found {len(all_files)} files: {all_files}")
    dataframes = {}
    for filepath in all_files:
        name = os.path.basename(filepath).replace(f".{file_format}", "")
        if file_format == "csv":
            df = pd.read_csv(filepath)
            first_col = str(df.columns[0])
            is_headerless = (
                first_col not in ["user_id", "ad_campaign_id", "product_category", "impressions", "clicks", "purchase_amount"]
                and len(df.columns) == len(CLEANROOMS_COLUMNS)
            )
            if is_headerless:
                df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
            elif len(df.columns) == len(CLEANROOMS_COLUMNS) - 1:
                df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
        else:
            df = pd.read_parquet(filepath)
        dataframes[name] = df
        logger.info(f"  Loaded {name}: {df.shape}")
    return dataframes


def engineer_features(dataframes):
    pre_joined_df = None
    for name, df in dataframes.items():
        has_adv = "impressions" in df.columns or "ad_campaign_id" in df.columns
        has_ret = "purchase_amount" in df.columns or "product_category" in df.columns
        if has_adv and has_ret:
            pre_joined_df = df
            break

    if pre_joined_df is not None:
        return _engineer_features_prejoined(pre_joined_df)
    return _engineer_features_separate(dataframes)


def _engineer_features_prejoined(df):
    df["click_through_rate"] = df["clicks"] / df["impressions"].clip(lower=1)
    df["time_per_click"] = df["time_spent_seconds"] / df["clicks"].clip(lower=1)
    feature_cols = [
        "impressions", "clicks", "time_spent_seconds", "click_through_rate", "time_per_click",
        "purchase_amount", "purchase_count", "site_visits", "days_since_last_purchase",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    y = df["converted"] if "converted" in df.columns else (df["purchase_amount"] > 0).astype(int)
    logger.info(f"Features shape: {X.shape}, Target dist:\n{y.value_counts().to_string()}")
    return X, y, feature_cols, None


def _engineer_features_separate(dataframes):
    advertiser_df = retailer_df = None
    for name, df in dataframes.items():
        if "ad_campaign_id" in df.columns or "impressions" in df.columns:
            advertiser_df = df
        elif "product_category" in df.columns or "purchase_amount" in df.columns:
            retailer_df = df
    if advertiser_df is None or retailer_df is None:
        raise ValueError(f"Could not identify both datasets. Columns: {[list(df.columns) for df in dataframes.values()]}")

    adv_agg = advertiser_df.groupby("user_id").agg(
        total_impressions=("impressions", "sum"), total_clicks=("clicks", "sum"),
        total_time_spent=("time_spent_seconds", "sum"), num_campaigns=("ad_campaign_id", "nunique"),
        num_devices=("device_type", "nunique"), avg_impressions=("impressions", "mean"),
        avg_clicks=("clicks", "mean"),
    ).reset_index()
    adv_agg["click_through_rate"] = adv_agg["total_clicks"] / adv_agg["total_impressions"].clip(lower=1)
    adv_agg["avg_time_per_click"] = adv_agg["total_time_spent"] / adv_agg["total_clicks"].clip(lower=1)

    ret_agg = retailer_df.groupby("user_id").agg(
        total_purchase_amount=("purchase_amount", "sum"), total_purchase_count=("purchase_count", "sum"),
        num_categories=("product_category", "nunique"), total_site_visits=("site_visits", "sum"),
        min_days_since_purchase=("days_since_last_purchase", "min"), avg_purchase_amount=("purchase_amount", "mean"),
    ).reset_index()
    if "converted" in retailer_df.columns:
        target = retailer_df.groupby("user_id")["converted"].max().reset_index()
        ret_agg = ret_agg.merge(target, on="user_id")
    else:
        ret_agg["converted"] = (ret_agg["total_purchase_amount"] > 0).astype(int)

    merged = adv_agg.merge(ret_agg, on="user_id", how="inner")
    feature_cols = [
        "total_impressions", "total_clicks", "total_time_spent", "num_campaigns", "num_devices",
        "avg_impressions", "avg_clicks", "click_through_rate", "avg_time_per_click",
        "total_site_visits", "num_categories",
    ]
    X = merged[feature_cols].fillna(0)
    y = merged["converted"]
    logger.info(f"Merged: {merged.shape[0]} users, Features: {X.shape}")
    return X, y, feature_cols, merged["user_id"]


def train_model(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                       learning_rate=args.learning_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_samples": X_train.shape[0], "test_samples": X_test.shape[0], "n_features": X_train.shape[1],
    }
    importance = dict(zip(X.columns, [round(float(v), 4) for v in model.feature_importances_]))
    metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))
    logger.info(f"Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'feature_importance'}, indent=2)}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return model, metrics


def save_artifacts(model, metrics, feature_cols, model_dir, output_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Artifacts saved to {model_dir} and {output_dir}")


def main():
    args = parse_args()
    logger.info(f"Arguments: {vars(args)}")
    try:
        sm_vars = {k: v for k, v in os.environ.items() if k.startswith("SM_") or k.startswith("FILE_")}
        logger.info(f"SageMaker env vars: {json.dumps(sm_vars, indent=2)}")
        dataframes = load_data(args.train_dir, args.train_file_format)
        X, y, feature_cols, user_ids = engineer_features(dataframes)
        model, metrics = train_model(X, y, args)
        save_artifacts(model, metrics, feature_cols, args.model_dir, args.output_dir)
        logger.info("Training completed successfully.")
    except Exception as e:
        failure_path = "/opt/ml/output/failure"
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(str(e)[:1024])
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
