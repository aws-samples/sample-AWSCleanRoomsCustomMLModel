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
AWS_ACCOUNT_ID = "CHANGE_ME"       # e.g. "123456789012"
AWS_REGION     = "CHANGE_ME"       # e.g. "eu-north-1", "us-east-1"

# ─── DERIVED (no need to change) ──────────────────────────
BUCKET         = f"cleanrooms-ml-demo-{AWS_ACCOUNT_ID}"
OUTPUT_BUCKET  = f"cleanrooms-ml-output-{AWS_ACCOUNT_ID}"
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


def validate():
    """Call this at the start of any script to catch misconfiguration early."""
    errors = []
    if AWS_ACCOUNT_ID == "CHANGE_ME" or not AWS_ACCOUNT_ID.isdigit() or len(AWS_ACCOUNT_ID) != 12:
        errors.append(f"AWS_ACCOUNT_ID must be a 12-digit number, got: '{AWS_ACCOUNT_ID}'")
    if AWS_REGION == "CHANGE_ME" or not AWS_REGION:
        errors.append(f"AWS_REGION must be set, got: '{AWS_REGION}'")
    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR — edit config.py")
        print("=" * 60)
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
