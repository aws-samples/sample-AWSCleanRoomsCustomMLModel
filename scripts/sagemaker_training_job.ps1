# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Launch a SageMaker Training Job using pre-built scikit-learn container
# Reads config from config.py

$config = python -c "import sys; sys.path.insert(0,'.'); from config import *; validate(); print(f'{AWS_ACCOUNT_ID} {AWS_REGION}')"
$parts = $config -split ' '
$ACCOUNT_ID = $parts[0]
$REGION = $parts[1]
$BUCKET = "cleanrooms-ml-demo-$ACCOUNT_ID"
$JOB_NAME = "cleanrooms-propensity-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$ROLE_NAME = "cleanrooms-ml-demo-sagemaker-role"

# Pre-built SageMaker scikit-learn 1.2 container
$SKLEARN_IMAGE = "662702820516.dkr.ecr.$REGION.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

Write-Output "Job Name: $JOB_NAME"
Write-Output "Image:    $SKLEARN_IMAGE"
Write-Output "Bucket:   s3://$BUCKET"
Write-Output ""

# Package training source code
Write-Output "Packaging training source code..."
$SOURCE_DIR = "containers/training"
$TAR_FILE = "scripts/sourcedir.tar.gz"

python -c @"
import tarfile, os
with tarfile.open('$($TAR_FILE -replace '\\','/')', 'w:gz') as tar:
    tar.add('$($SOURCE_DIR -replace '\\','/')/train.py', arcname='train.py')
print('Created sourcedir.tar.gz')
"@

$SOURCE_S3 = "s3://$BUCKET/sagemaker-source/sourcedir.tar.gz"
aws s3 cp $TAR_FILE $SOURCE_S3 --region $REGION
Write-Output "Source uploaded to: $SOURCE_S3"

# Create SageMaker execution role if needed
$ROLE_ARN = (aws iam get-role --role-name $ROLE_NAME --query "Role.Arn" --output text 2>$null)

if (-not $ROLE_ARN -or $ROLE_ARN -eq "None") {
    Write-Output "Creating SageMaker execution role..."
    $TRUST_POLICY = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    [System.IO.File]::WriteAllText("$PWD/scripts/trust-policy.json", $TRUST_POLICY, [System.Text.UTF8Encoding]::new($false))
    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://scripts/trust-policy.json --region $REGION
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" --region $REGION
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess" --region $REGION
    Start-Sleep -Seconds 15
    $ROLE_ARN = (aws iam get-role --role-name $ROLE_NAME --query "Role.Arn" --output text)
}

Write-Output "Role ARN: $ROLE_ARN"

$TRAINING_JOB = @"
{
    "TrainingJobName": "$JOB_NAME",
    "AlgorithmSpecification": {"TrainingImage": "$SKLEARN_IMAGE", "TrainingInputMode": "File"},
    "RoleArn": "$ROLE_ARN",
    "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://$BUCKET/data/", "S3DataDistributionType": "FullyReplicated"}}, "ContentType": "text/csv", "InputMode": "File"}],
    "OutputDataConfig": {"S3OutputPath": "s3://$BUCKET/sagemaker-output/"},
    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.4xlarge", "VolumeSizeInGB": 10},
    "EnableManagedSpotTraining": true,
    "StoppingCondition": {"MaxRuntimeInSeconds": 600, "MaxWaitTimeInSeconds": 1200},
    "HyperParameters": {"n_estimators": "100", "max_depth": "5", "learning_rate": "0.1", "sagemaker_program": "train.py", "sagemaker_submit_directory": "\"$SOURCE_S3\""}
}
"@

[System.IO.File]::WriteAllText("$PWD/scripts/training_job.json", $TRAINING_JOB, [System.Text.UTF8Encoding]::new($false))

Write-Output "Submitting SageMaker training job..."
aws sagemaker create-training-job --cli-input-json file://scripts/training_job.json --region $REGION

Write-Output ""
Write-Output "Job submitted: $JOB_NAME"
Write-Output "Console: https://$REGION.console.aws.amazon.com/sagemaker/home?region=$REGION#/jobs/$JOB_NAME"
