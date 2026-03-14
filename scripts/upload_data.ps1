# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Upload synthetic data to S3 for Clean Rooms ML demo
# Reads ACCOUNT_ID and REGION from config.py via a quick Python call

$config = python -c "import sys; sys.path.insert(0,'.'); from config import *; validate(); print(f'{AWS_ACCOUNT_ID} {AWS_REGION}')"
$parts = $config -split ' '
$ACCOUNT_ID = $parts[0]
$REGION = $parts[1]
$BUCKET_NAME = "cleanrooms-ml-demo-$ACCOUNT_ID"

Write-Output "Account: $ACCOUNT_ID"
Write-Output "Region:  $REGION"
Write-Output "Bucket:  $BUCKET_NAME"
Write-Output ""

# Create bucket
Write-Output "Creating S3 bucket..."
aws s3api create-bucket `
    --bucket $BUCKET_NAME `
    --region $REGION `
    --create-bucket-configuration LocationConstraint=$REGION

# Upload advertiser data (Party A)
Write-Output "Uploading advertiser data..."
aws s3 cp data/advertiser_engagement.csv "s3://$BUCKET_NAME/advertiser/advertiser_engagement.csv" --region $REGION

# Upload retailer data (Party B)
Write-Output "Uploading retailer data..."
aws s3 cp data/retailer_purchases.csv "s3://$BUCKET_NAME/retailer/retailer_purchases.csv" --region $REGION

# Upload both CSVs under data/ prefix for SageMaker training channel
Write-Output "Uploading combined data for SageMaker training..."
aws s3 cp data/advertiser_engagement.csv "s3://$BUCKET_NAME/data/advertiser_engagement.csv" --region $REGION
aws s3 cp data/retailer_purchases.csv "s3://$BUCKET_NAME/data/retailer_purchases.csv" --region $REGION

# Create output bucket
$OUTPUT_BUCKET = "cleanrooms-ml-output-$ACCOUNT_ID"
Write-Output "Creating output bucket: $OUTPUT_BUCKET"
aws s3api create-bucket `
    --bucket $OUTPUT_BUCKET `
    --region $REGION `
    --create-bucket-configuration LocationConstraint=$REGION

# Verify
Write-Output ""
Write-Output "Verifying uploads..."
aws s3 ls "s3://$BUCKET_NAME/" --recursive --region $REGION

Write-Output ""
Write-Output "Done!"
