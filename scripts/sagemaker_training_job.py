# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Launch a SageMaker AI Training Job using pre-built scikit-learn container.
Reads config from config.py.
"""
import sys, os, tarfile, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3
from datetime import datetime

iam = boto3.client("iam", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)
sm = boto3.client("sagemaker", region_name=AWS_REGION)

ROLE_NAME = "cleanrooms-ml-demo-sagemaker-role"
SKLEARN_IMAGE = f"662702820516.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

project_root = os.path.join(os.path.dirname(__file__), "..")


def log(msg):
    print(f"  → {msg}")


def package_source():
    """Create sourcedir.tar.gz with train.py."""
    tar_path = os.path.join(project_root, "scripts", "sourcedir.tar.gz")
    train_py = os.path.join(project_root, "containers", "training", "train.py")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(train_py, arcname="train.py")
    log("Created sourcedir.tar.gz")

    s3_key = "sagemaker-source/sourcedir.tar.gz"
    s3.upload_file(tar_path, BUCKET, s3_key)
    source_s3 = f"s3://{BUCKET}/{s3_key}"
    log(f"Uploaded to {source_s3}")
    return source_s3


def ensure_role():
    """Create SageMaker AI execution role if it doesn't exist."""
    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        log(f"Role exists: {ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        log(f"Creating role: {ROLE_NAME}")
        trust = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        })
        resp = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=trust)
        role_arn = resp["Role"]["Arn"]

        # Scoped SageMaker AI permissions instead of AmazonSageMakerFullAccess
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName=f"{ROLE_NAME}-sagemaker-policy",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "SageMakerTrainingAccess",
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateTrainingJob",
                            "sagemaker:DescribeTrainingJob",
                            "sagemaker:StopTrainingJob",
                            "sagemaker:ListTags",
                        ],
                        "Resource": f"arn:aws:sagemaker:{AWS_REGION}:{AWS_ACCOUNT_ID}:training-job/cleanrooms-*",
                    },
                    {
                        "Sid": "CloudWatchLogs",
                        "Effect": "Allow",
                        "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                        "Resource": f"arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_ID}:log-group:/aws/sagemaker/*",
                    },
                    {
                        "Sid": "CloudWatchMetrics",
                        "Effect": "Allow",
                        # cloudwatch:PutMetricData does not support resource-level permissions
                        "Action": ["cloudwatch:PutMetricData"],
                        "Resource": "*",
                    },
                    {
                        "Sid": "ECRPull",
                        "Effect": "Allow",
                        "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
                        "Resource": f"arn:aws:ecr:{AWS_REGION}:662702820516:repository/sagemaker-scikit-learn",
                    },
                    {
                        "Sid": "ECRAuth",
                        "Effect": "Allow",
                        # ecr:GetAuthorizationToken does not support resource-level permissions
                        "Action": ["ecr:GetAuthorizationToken"],
                        "Resource": "*",
                    },
                ],
            }),
        )

        # Scoped S3 permissions instead of AmazonS3FullAccess
        iam.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName=f"{ROLE_NAME}-s3-policy",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "S3DataAccess",
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject"],
                        "Resource": [
                            f"arn:aws:s3:::{BUCKET}/data/*",
                            f"arn:aws:s3:::{BUCKET}/sagemaker-source/*",
                            f"arn:aws:s3:::{BUCKET}/sagemaker-output/*",
                        ],
                        "Condition": {"Bool": {"aws:SecureTransport": "true"}},
                    },
                    {
                        "Sid": "S3BucketList",
                        "Effect": "Allow",
                        "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                        "Resource": f"arn:aws:s3:::{BUCKET}",
                        "Condition": {"Bool": {"aws:SecureTransport": "true"}},
                    },
                ],
            }),
        )
        log("Attached scoped inline policies (SageMaker AI + S3)")
        log("Waiting 15s for role propagation...")
        time.sleep(15)
    return role_arn


def main():
    job_name = f"cleanrooms-propensity-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("=" * 60)
    print("SageMaker AI Training Job")
    print("=" * 60)
    print(f"Job Name: {job_name}")
    print(f"Image:    {SKLEARN_IMAGE}")
    print(f"Bucket:   s3://{BUCKET}\n")

    source_s3 = package_source()
    role_arn = ensure_role()

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": SKLEARN_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=[{
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
            "InputMode": "File",
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/sagemaker-output/"},
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": "ml.m5.4xlarge",
            "VolumeSizeInGB": 10,
        },
        EnableManagedSpotTraining=True,
        StoppingCondition={
            "MaxRuntimeInSeconds": 600,
            "MaxWaitTimeInSeconds": 1200,
        },
        HyperParameters={
            "n_estimators": "100",
            "max_depth": "5",
            "learning_rate": "0.1",
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": f'"{source_s3}"',
        },
    )

    print(f"\nJob submitted: {job_name}")
    print(f"Console: https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs/{job_name}")


if __name__ == "__main__":
    main()
