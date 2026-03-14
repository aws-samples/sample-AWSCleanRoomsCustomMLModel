# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML — Create ML Input Channels, Train Model, Run Inference.
Reads config from config.py. Assumes setup_cleanrooms.py has been run.
"""
import sys, os, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3

cr = crml = sts = None

def init_clients():
    global cr, crml, sts
    session = boto3.Session(region_name=AWS_REGION)
    cr = session.client("cleanrooms")
    crml = session.client("cleanroomsml")
    sts = session.client("sts")

def log(msg):
    print(f"  → {msg}")

def get_membership_and_collab():
    memberships = cr.list_memberships(status="ACTIVE")
    for m in memberships.get("membershipSummaries", []):
        if PREFIX in m.get("collaborationName", ""):
            return m["id"], m["collaborationId"]
    print("ERROR: No active membership found. Run setup_cleanrooms.py first.")
    sys.exit(1)

def get_algo_association(membership_id):
    assocs = crml.list_configured_model_algorithm_associations(membershipIdentifier=membership_id)
    for a in assocs.get("configuredModelAlgorithmAssociations", []):
        if "propensity" in a.get("name", ""):
            return a["configuredModelAlgorithmAssociationArn"]
    print("ERROR: No model algorithm association found.")
    sys.exit(1)

def get_configured_table_associations(membership_id):
    assocs = cr.list_configured_table_associations(membershipIdentifier=membership_id)
    result = {}
    for a in assocs.get("configuredTableAssociationSummaries", []):
        if "advertiser" in a["name"]:
            result["advertiser"] = a["arn"]
        elif "retailer" in a["name"]:
            result["retailer"] = a["arn"]
    if len(result) != 2:
        print(f"ERROR: Expected 2 table associations, found {len(result)}")
        sys.exit(1)
    return result


def create_ml_input_channel(membership_id, collab_id, table_assocs, channel_name, channel_purpose):
    print(f"\n  Creating ML input channel: {channel_name} ({channel_purpose})...")
    existing = crml.list_ml_input_channels(membershipIdentifier=membership_id)
    for ch in existing.get("mlInputChannelsList", []):
        if ch.get("name", "").startswith(channel_name):
            status = ch.get("status", "")
            if status in ("ACTIVE", "CREATE_PENDING", "CREATE_IN_PROGRESS"):
                log(f"ML input channel already exists: {ch['name']} ({status})")
                return ch["mlInputChannelArn"]

    ts = datetime.datetime.now().strftime("%H%M%S")
    channel_name = f"{channel_name}-{ts}"

    resp = crml.create_ml_input_channel(
        membershipIdentifier=membership_id,
        configuredModelAlgorithmAssociations=[get_algo_association(membership_id)],
        name=channel_name,
        description=f"{channel_purpose} input channel for propensity model",
        inputChannel={
            "dataSource": {
                "protectedQueryInputParameters": {
                    "sqlParameters": {
                        "queryString": (
                            'SELECT DISTINCT a.ad_campaign_id, a.impressions, a.clicks, '
                            'a.time_spent_seconds, a.device_type, a.event_date, '
                            'r.product_category, r.purchase_amount, r.purchase_count, '
                            'r.site_visits, r.days_since_last_purchase, r.last_purchase_date, '
                            'r.converted '
                            'FROM advertiser_association a '
                            'INNER JOIN retailer_association r ON a.user_id = r.user_id'
                        ),
                    }
                }
            },
            "roleArn": f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{ROLE_QUERY_RUNNER}",
        },
        retentionInDays=30,
    )
    log(f"Created ML input channel: {channel_name}")
    return resp["mlInputChannelArn"]


def wait_for_ml_input_channel(membership_id, channel_arn, channel_name):
    print(f"  Waiting for {channel_name} to become ACTIVE...")
    while True:
        resp = crml.get_ml_input_channel(mlInputChannelArn=channel_arn, membershipIdentifier=membership_id)
        status = resp["status"]
        if status == "ACTIVE":
            log(f"{channel_name} is ACTIVE")
            return True
        elif status in ("CREATE_FAILED", "CANCELLED"):
            log(f"{channel_name} failed: {status}")
            log(f"Details: {resp.get('statusDetails', {}).get('message', 'No details')}")
            return False
        print(f"    Status: {status}...", end="\r")
        time.sleep(30)


def create_trained_model(membership_id, collab_id, algo_assoc_arn, training_channel_arn):
    print("\n  Creating trained model...")
    existing = crml.list_trained_models(membershipIdentifier=membership_id)
    for m in existing.get("trainedModels", []):
        if "propensity" in m.get("name", ""):
            status = m.get("status", "")
            if status in ("ACTIVE", "CREATE_PENDING", "CREATE_IN_PROGRESS"):
                log(f"Trained model already exists: {m['name']} ({status})")
                return m["trainedModelArn"]

    ts = datetime.datetime.now().strftime("%H%M%S")
    resp = crml.create_trained_model(
        membershipIdentifier=membership_id,
        name=f"{PREFIX}-propensity-trained-{ts}",
        description="Customer propensity scoring trained model",
        configuredModelAlgorithmAssociationArn=algo_assoc_arn,
        resourceConfig={"instanceCount": 1, "instanceType": "ml.m5.2xlarge", "volumeSizeInGB": 30},
        dataChannels=[{"mlInputChannelArn": training_channel_arn, "channelName": "training"}],
    )
    log(f"Started training: {resp['trainedModelArn']}")
    return resp["trainedModelArn"]


def wait_for_trained_model(membership_id, model_arn):
    print("  Waiting for training to complete...")
    while True:
        resp = crml.get_trained_model(trainedModelArn=model_arn, membershipIdentifier=membership_id)
        status = resp["status"]
        if status == "ACTIVE":
            log("Training SUCCEEDED — model is ACTIVE")
            return True
        elif status in ("CREATE_FAILED", "CANCELLED"):
            log(f"Training failed: {status}")
            log(f"Details: {resp.get('statusDetails', {}).get('message', 'No details')}")
            return False
        print(f"    Status: {status}...", end="\r")
        time.sleep(30)


def run_inference_job(membership_id, model_arn, inference_channel_arn):
    print("\n  Running inference job...")
    ts = datetime.datetime.now().strftime("%H%M%S")
    resp = crml.start_trained_model_inference_job(
        membershipIdentifier=membership_id,
        name=f"{PREFIX}-propensity-inference-{ts}",
        trainedModelArn=model_arn,
        resourceConfig={"instanceCount": 1, "instanceType": "ml.m5.2xlarge"},
        dataSource={"mlInputChannelArn": inference_channel_arn},
        outputConfiguration={"accept": "text/csv", "members": [{"accountId": AWS_ACCOUNT_ID}]},
    )
    log(f"Started inference job: {resp['trainedModelInferenceJobArn']}")
    return resp["trainedModelInferenceJobArn"]


def wait_for_inference_job(membership_id, job_arn):
    print("  Waiting for inference to complete...")
    while True:
        resp = crml.get_trained_model_inference_job(trainedModelInferenceJobArn=job_arn, membershipIdentifier=membership_id)
        status = resp["status"]
        if status == "ACTIVE":
            log("Inference SUCCEEDED")
            return True
        elif status in ("CREATE_FAILED", "CANCELLED"):
            log(f"Inference failed: {status}")
            log(f"Details: {resp.get('statusDetails', {}).get('message', 'No details')}")
            return False
        print(f"    Status: {status}...", end="\r")
        time.sleep(30)


def main():
    print("=" * 60)
    print("AWS Clean Rooms ML — Train & Inference")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")

    init_clients()
    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    membership_id, collab_id = get_membership_and_collab()
    log(f"Membership: {membership_id}, Collaboration: {collab_id}")

    algo_assoc_arn = get_algo_association(membership_id)
    table_assocs = get_configured_table_associations(membership_id)

    # Step 1: Training channel
    print("\n[1/5] Creating training ML input channel...")
    training_arn = create_ml_input_channel(membership_id, collab_id, table_assocs, f"{PREFIX}-training-channel", "Training")
    print("\n[2/5] Waiting for training channel...")
    if not wait_for_ml_input_channel(membership_id, training_arn, "training"):
        sys.exit(1)

    # Step 2: Inference channel
    print("\n[3/5] Creating inference ML input channel...")
    inference_arn = create_ml_input_channel(membership_id, collab_id, table_assocs, f"{PREFIX}-inference-channel", "Inference")
    if not wait_for_ml_input_channel(membership_id, inference_arn, "inference"):
        sys.exit(1)

    # Step 3: Train model
    print("\n[4/5] Creating trained model...")
    model_arn = create_trained_model(membership_id, collab_id, algo_assoc_arn, training_arn)
    if not wait_for_trained_model(membership_id, model_arn):
        sys.exit(1)

    # Step 4: Run inference
    print("\n[5/5] Running inference job...")
    job_arn = run_inference_job(membership_id, model_arn, inference_arn)
    if not wait_for_inference_job(membership_id, job_arn):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nInference output: s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/")
    print(f"Console: https://{AWS_REGION}.console.aws.amazon.com/cleanrooms/home?region={AWS_REGION}#/collaborations/{collab_id}")


if __name__ == "__main__":
    main()
