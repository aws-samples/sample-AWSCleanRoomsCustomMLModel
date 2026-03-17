# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Build and push training/inference containers to ECR using AWS CodeBuild.
No local Docker required. Reads config from config.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3, json, time, zipfile, io

PROJECT_NAME = "cleanrooms-ml-demo-build"
CB_ROLE_NAME = "cleanrooms-ml-demo-codebuild-role"
SOURCE_KEY = "codebuild/source.zip"

sts = boto3.client("sts", region_name=AWS_REGION)
iam = boto3.client("iam", region_name=AWS_REGION)
cb = boto3.client("codebuild", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)
ecr = boto3.client("ecr", region_name=AWS_REGION)

def log(msg):
    print(f"  → {msg}")

def ensure_ecr_repos():
    for repo in ["cleanrooms-ml-demo-training", "cleanrooms-ml-demo-inference"]:
        try:
            ecr.create_repository(repositoryName=repo)
            log(f"Created ECR repo: {repo}")
        except ecr.exceptions.RepositoryAlreadyExistsException:
            log(f"ECR repo exists: {repo}")

def create_codebuild_role():
    trust = json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole"}]})
    try:
        resp = iam.create_role(RoleName=CB_ROLE_NAME, AssumeRolePolicyDocument=trust)
        role_arn = resp["Role"]["Arn"]
        log(f"Created role: {CB_ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{CB_ROLE_NAME}"
        log(f"Role exists: {CB_ROLE_NAME}")

    iam.put_role_policy(RoleName=CB_ROLE_NAME, PolicyName=f"{PROJECT_NAME}-policy", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "CloudWatchLogs",
             "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": f"arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_ID}:log-group:/aws/codebuild/*"},
            {"Sid": "S3SourceAccess",
             "Effect": "Allow", "Action": ["s3:GetObject", "s3:GetBucketLocation"],
             "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/codebuild/*"]},
            {"Sid": "ECRAuthToken",
             "Effect": "Allow",
             # ecr:GetAuthorizationToken does not support resource-level permissions
             "Action": ["ecr:GetAuthorizationToken"],
             "Resource": "*"},
            {"Sid": "ECRPushPull",
             "Effect": "Allow",
             "Action": [
                "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage", "ecr:PutImage", "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"],
             "Resource": [
                f"arn:aws:ecr:{AWS_REGION}:{AWS_ACCOUNT_ID}:repository/cleanrooms-ml-demo-training",
                f"arn:aws:ecr:{AWS_REGION}:{AWS_ACCOUNT_ID}:repository/cleanrooms-ml-demo-inference"]},
            {"Sid": "ECRPullSageMakerAIDLC",
             "Effect": "Allow",
             "Action": [
                "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"],
             "Resource": [
                f"arn:aws:ecr:{AWS_REGION}:{SAGEMAKER_REGISTRY}:repository/pytorch-training",
                f"arn:aws:ecr:{AWS_REGION}:{SAGEMAKER_REGISTRY}:repository/pytorch-inference"]},
        ],
    }))
    time.sleep(10)
    return role_arn


def upload_source():
    buf = io.BytesIO()
    # Zip from the project root (parent of scripts/)
    project_root = os.path.join(os.path.dirname(__file__), "..")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(os.path.join(project_root, "containers")):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, project_root)
                zf.write(filepath, arcname)
        zf.write(os.path.join(project_root, "buildspec.yml"), "buildspec.yml")
        zf.write(os.path.join(project_root, "pyproject.toml"), "pyproject.toml")
        zf.write(os.path.join(project_root, "uv.lock"), "uv.lock")
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=SOURCE_KEY, Body=buf.read())
    log(f"Uploaded source to s3://{BUCKET}/{SOURCE_KEY}")


def create_or_update_project(role_arn):
    project_config = dict(
        name=PROJECT_NAME,
        description="Build AWS Clean Rooms ML demo containers",
        source={"type": "S3", "location": f"{BUCKET}/{SOURCE_KEY}"},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={
            "type": "LINUX_CONTAINER",
            "image": "aws/codebuild/standard:7.0",
            "computeType": "BUILD_GENERAL1_MEDIUM",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "ACCOUNT_ID", "value": AWS_ACCOUNT_ID, "type": "PLAINTEXT"},
                {"name": "REGION", "value": AWS_REGION, "type": "PLAINTEXT"},
                {"name": "SAGEMAKER_REGISTRY", "value": SAGEMAKER_REGISTRY, "type": "PLAINTEXT"},
            ],
        },
        serviceRole=role_arn,
    )
    try:
        cb.create_project(**project_config)
        log(f"Created CodeBuild project: {PROJECT_NAME}")
    except cb.exceptions.ResourceAlreadyExistsException:
        cb.update_project(**project_config)
        log(f"Updated CodeBuild project: {PROJECT_NAME}")


def run_build():
    resp = cb.start_build(projectName=PROJECT_NAME)
    build_id = resp["build"]["id"]
    log(f"Started build: {build_id}")
    print("\n  Waiting for build to complete...")
    while True:
        status = cb.batch_get_builds(ids=[build_id])["builds"][0]
        phase = status.get("currentPhase", "UNKNOWN")
        build_status = status["buildStatus"]
        if build_status == "IN_PROGRESS":
            print(f"    Phase: {phase}...", end="\r")
            time.sleep(15)
        else:
            print()
            if build_status == "SUCCEEDED":
                log("Build SUCCEEDED")
            else:
                log(f"Build {build_status}")
                log_group = status.get("logs", {}).get("groupName", "")
                log_stream = status.get("logs", {}).get("streamName", "")
                if log_group:
                    log(f"Logs: {log_group}/{log_stream}")
            return build_status


def main():
    print("=" * 60)
    print("Build Containers via CodeBuild (no local Docker needed)")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")

    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    ensure_ecr_repos()
    role_arn = create_codebuild_role()
    upload_source()
    create_or_update_project(role_arn)
    status = run_build()

    if status == "SUCCEEDED":
        print(f"\nContainers pushed to ECR:")
        print(f"  Training:  {TRAINING_IMAGE}")
        print(f"  Inference: {INFERENCE_IMAGE}")
        print(f"\nNext: python scripts/setup_cleanrooms.py")
    else:
        print("\nBuild failed. Check CloudWatch logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
