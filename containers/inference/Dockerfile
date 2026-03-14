# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Inference container for Customer Propensity Scoring
# Uses SageMaker AI inference base image as required by Clean Rooms ML docs
ARG SAGEMAKER_REGISTRY=763104351884
ARG AWS_REGION=us-east-1
FROM ${SAGEMAKER_REGISTRY}.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker

ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn joblib

COPY serve.py /opt/ml/code/serve.py
COPY inference_handler.py /opt/ml/code/inference_handler.py

RUN chmod +x /opt/ml/code/serve.py

ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
