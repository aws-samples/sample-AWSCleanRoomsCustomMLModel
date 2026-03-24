# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
═══════════════════════════════════════════════════════════════
  AWS Clean Rooms ML Custom Model Demo — Configuration
═══════════════════════════════════════════════════════════════

  Fill in your AWS Account ID and Region below.
  All scripts in this demo read from this file.

  Usage:
    1. Set AWS_ACCOUNT_ID and AWS_REGION
    2. Run the scripts in order (see README)
"""

# ─── REQUIRED: Set these to your values ───────────────────
# Edit the values below directly. These are the authoritative settings.
# Environment variables AWS_ACCOUNT_ID / AWS_REGION are only used as
# fallback when the placeholder values below have not been changed.
import os as _os_cfg

_ACCOUNT_DEFAULT = "123456789012"
_REGION_DEFAULT  = "eu-west-2"
_EMAIL_DEFAULT   = "your-email@example.com"

AWS_ACCOUNT_ID        = _ACCOUNT_DEFAULT if _ACCOUNT_DEFAULT != "123456789012" else _os_cfg.environ.get("AWS_ACCOUNT_ID", "123456789012")
AWS_REGION            = _REGION_DEFAULT  if _REGION_DEFAULT  != "us-east-1"    else _os_cfg.environ.get("AWS_REGION",     "us-east-1")

# ─── OPTIONAL: Required only for scripts/create_dashboard.py ──
# Email address for QuickSight account registration and admin user.
# Must be a valid address — QuickSight sends subscription notifications to it.
QS_NOTIFICATION_EMAIL = _EMAIL_DEFAULT   if _EMAIL_DEFAULT   != "your-email@example.com" else _os_cfg.environ.get("QS_NOTIFICATION_EMAIL", "your-email@example.com")

# ─── RUN ID (auto-generated, ensures unique bucket names) ─
import os as _os
from datetime import datetime as _dt

_RUN_ID_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".run_id")

def _get_or_create_run_id():
    """Return a short unique suffix for bucket names.

    Generated once and persisted to .run_id so every script in the same
    demo run shares the same buckets.  Delete .run_id to start fresh.
    """
    if _os.path.exists(_RUN_ID_FILE):
        return open(_RUN_ID_FILE).read().strip()
    run_id = _dt.utcnow().strftime("%Y%m%d%H%M")
    with open(_RUN_ID_FILE, "w") as f:
        f.write(run_id)
    return run_id

RUN_ID = _get_or_create_run_id()

# ─── DERIVED (no need to change) ──────────────────────────
BUCKET         = f"cleanrooms-ml-demo-{AWS_ACCOUNT_ID}-{RUN_ID}"
OUTPUT_BUCKET  = f"cleanrooms-ml-output-{AWS_ACCOUNT_ID}-{RUN_ID}"
PREFIX         = "cleanrooms-ml-demo"

# ECR image URIs
TRAINING_IMAGE  = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cleanrooms-ml-demo-training:latest"
INFERENCE_IMAGE = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cleanrooms-ml-demo-inference:latest"

# SageMaker DLC registry (same account across all regions)
SAGEMAKER_REGISTRY = "763104351884"
SAGEMAKER_TRAINING_BASE  = f"{SAGEMAKER_REGISTRY}.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker"
SAGEMAKER_INFERENCE_BASE = f"{SAGEMAKER_REGISTRY}.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker"

# Glue
GLUE_DB            = "cleanrooms_ml_demo"
ADVERTISER_TABLE   = "advertiser_engagement"
RETAILER_TABLE     = "retailer_purchases"

# IAM role names
ROLE_DATA_PROVIDER  = f"{PREFIX}-data-provider-role"
ROLE_MODEL_PROVIDER = f"{PREFIX}-model-provider-role"
ROLE_ML_CONFIG      = f"{PREFIX}-ml-config-role"
ROLE_QUERY_RUNNER   = f"{PREFIX}-query-runner-role"


def validate(require_qs_email=False):
    """Call this at the start of any script to catch misconfiguration early."""
    errors = []
    if AWS_ACCOUNT_ID == "CHANGE_ME" or not AWS_ACCOUNT_ID.isdigit() or len(AWS_ACCOUNT_ID) != 12:
        errors.append(f"AWS_ACCOUNT_ID must be a 12-digit number, got: '{AWS_ACCOUNT_ID}'")
    if AWS_REGION == "CHANGE_ME" or not AWS_REGION:
        errors.append(f"AWS_REGION must be set, got: '{AWS_REGION}'")
    if require_qs_email and QS_NOTIFICATION_EMAIL == "your-email@example.com":
        errors.append("QS_NOTIFICATION_EMAIL must be set to a real email address in config.py "
                      "(required for QuickSight account registration)")
    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR — edit config.py")
        print("=" * 60)
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
