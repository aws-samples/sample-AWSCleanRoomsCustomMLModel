# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Build and push both containers to ECR (requires local Docker)
# Reads config from config.py

$config = python -c "import sys; sys.path.insert(0,'.'); from config import *; validate(); print(f'{AWS_ACCOUNT_ID} {AWS_REGION} {SAGEMAKER_REGISTRY}')"
$parts = $config -split ' '
$ACCOUNT_ID = $parts[0]
$REGION = $parts[1]
$SAGEMAKER_REG = $parts[2]
$ECR_ENDPOINT = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

$TRAINING_REPO = "cleanrooms-ml-demo-training"
$INFERENCE_REPO = "cleanrooms-ml-demo-inference"
$TAG = "latest"

Write-Output "Account:  $ACCOUNT_ID"
Write-Output "Region:   $REGION"
Write-Output ""

# Authenticate Docker with ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_ENDPOINT
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$SAGEMAKER_REG.dkr.ecr.$REGION.amazonaws.com"

# Training
aws ecr create-repository --repository-name $TRAINING_REPO --region $REGION 2>$null
docker build --build-arg AWS_REGION=$REGION --build-arg SAGEMAKER_REGISTRY=$SAGEMAKER_REG -t "$ECR_ENDPOINT/${TRAINING_REPO}:${TAG}" containers/training/
docker push "$ECR_ENDPOINT/${TRAINING_REPO}:${TAG}"

# Inference
aws ecr create-repository --repository-name $INFERENCE_REPO --region $REGION 2>$null
docker build --build-arg AWS_REGION=$REGION --build-arg SAGEMAKER_REGISTRY=$SAGEMAKER_REG -t "$ECR_ENDPOINT/${INFERENCE_REPO}:${TAG}" containers/inference/
docker push "$ECR_ENDPOINT/${INFERENCE_REPO}:${TAG}"

Write-Output ""
Write-Output "Training image:  $ECR_ENDPOINT/${TRAINING_REPO}:${TAG}"
Write-Output "Inference image: $ECR_ENDPOINT/${INFERENCE_REPO}:${TAG}"
