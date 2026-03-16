# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Training container for Customer Propensity Scoring
# Compatible with: local, SageMaker AI Training, AWS Clean Rooms ML
ARG SAGEMAKER_REGISTRY=763104351884
ARG AWS_REGION=us-east-1
FROM ${SAGEMAKER_REGISTRY}.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker

MAINTAINER cleanrooms-ml-demo

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn joblib

# Copy training script
COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
