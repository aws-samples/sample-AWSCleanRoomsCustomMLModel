# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Build and push training/inference containers to ECR using local Docker.
Reads config from config.py.
"""
import sys, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3

ecr = boto3.client("ecr", region_name=AWS_REGION)

ECR_ENDPOINT = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
TRAINING_REPO = "cleanrooms-ml-demo-training"
INFERENCE_REPO = "cleanrooms-ml-demo-inference"
TAG = "latest"

project_root = os.path.join(os.path.dirname(__file__), "..")


def log(msg):
    print(f"  → {msg}")


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def ensure_ecr_repo(repo_name):
    try:
        ecr.create_repository(repositoryName=repo_name)
        log(f"Created ECR repo: {repo_name}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        log(f"ECR repo exists: {repo_name}")


def docker_login(registry):
    token = subprocess.run(
        f"aws ecr get-login-password --region {AWS_REGION}",
        shell=True, capture_output=True, text=True
    )
    result = subprocess.run(
        f"docker login --username AWS --password-stdin {registry}",
        shell=True, input=token.stdout, capture_output=True, text=True
    )
    if result.returncode == 0:
        log(f"Docker login OK: {registry}")
    else:
        log(f"Docker login FAILED: {result.stderr.strip()}")
        sys.exit(1)


def build_and_push(repo_name, context_dir):
    image_tag = f"{ECR_ENDPOINT}/{repo_name}:{TAG}"
    run(
        f"docker build"
        f" --build-arg AWS_REGION={AWS_REGION}"
        f" --build-arg SAGEMAKER_REGISTRY={SAGEMAKER_REGISTRY}"
        f" -t {image_tag} {context_dir}",
        cwd=project_root
    )
    run(f"docker push {image_tag}")
    log(f"Pushed: {image_tag}")


def main():
    print("=" * 60)
    print("Build & Push Containers (local Docker)")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}\n")

    # Authenticate Docker with ECR
    docker_login(ECR_ENDPOINT)
    docker_login(f"{SAGEMAKER_REGISTRY}.dkr.ecr.{AWS_REGION}.amazonaws.com")

    # Create repos
    ensure_ecr_repo(TRAINING_REPO)
    ensure_ecr_repo(INFERENCE_REPO)

    # Build and push
    build_and_push(TRAINING_REPO, "containers/training/")
    build_and_push(INFERENCE_REPO, "containers/inference/")

    print(f"\nTraining image:  {ECR_ENDPOINT}/{TRAINING_REPO}:{TAG}")
    print(f"Inference image: {ECR_ENDPOINT}/{INFERENCE_REPO}:{TAG}")


if __name__ == "__main__":
    main()
