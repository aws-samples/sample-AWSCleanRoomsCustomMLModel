# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Upload synthetic data to S3 for Clean Rooms ML demo.
Creates source and output buckets, uploads CSVs.
Reads config from config.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3, json

s3 = boto3.client("s3", region_name=AWS_REGION)

def log(msg):
    print(f"  → {msg}")


def create_bucket(bucket_name):
    """Create an S3 bucket, handling the us-east-1 LocationConstraint quirk.

    Newer versions of botocore route us-east-1 requests to the regional
    endpoint (s3.us-east-1.amazonaws.com) which rejects both an omitted
    and an explicit 'us-east-1' LocationConstraint.  The workaround is to
    point the client at the regional endpoint and pass the constraint.
    For all other regions the standard approach works fine.
    """
    try:
        if AWS_REGION == "us-east-1":
            s3_us = boto3.client(
                "s3",
                region_name="us-east-1",
                endpoint_url="https://s3.us-east-1.amazonaws.com",
            )
            s3_us.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )
        log(f"Created bucket: {bucket_name}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e):
            log(f"Bucket already exists: {bucket_name}")
        else:
            raise

    # Block all public access
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    log(f"  Enabled Block Public Access on {bucket_name}")

    # Default encryption (SSE-S3)
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}],
        },
    )
    log(f"  Enabled default SSE-S3 encryption on {bucket_name}")

    # Versioning
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={"Status": "Enabled"},
    )
    log(f"  Enabled versioning on {bucket_name}")

    # Enforce TLS-only access via bucket policy
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "DenyInsecureTransport",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }],
        }),
    )
    log(f"  Enforced TLS-only bucket policy on {bucket_name}")


def upload_file(local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)
    log(f"Uploaded {local_path} → s3://{bucket}/{key}")


def main():
    print("=" * 60)
    print("Upload Data to S3")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Source bucket:  {BUCKET}")
    print(f"Output bucket:  {OUTPUT_BUCKET}")
    print()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    adv_csv = os.path.join(project_root, "data", "advertiser_engagement.csv")
    ret_csv = os.path.join(project_root, "data", "retailer_purchases.csv")

    # Create buckets
    create_bucket(BUCKET)
    create_bucket(OUTPUT_BUCKET)

    # Upload advertiser data (Party A)
    upload_file(adv_csv, BUCKET, "advertiser/advertiser_engagement.csv")

    # Upload retailer data (Party B)
    upload_file(ret_csv, BUCKET, "retailer/retailer_purchases.csv")

    # Upload both under data/ prefix for SageMaker training channel
    upload_file(adv_csv, BUCKET, "data/advertiser_engagement.csv")
    upload_file(ret_csv, BUCKET, "data/retailer_purchases.csv")

    # Verify
    print("\nVerifying uploads...")
    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        print(f"  {obj['Key']}  ({obj['Size']} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
