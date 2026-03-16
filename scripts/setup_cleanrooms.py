# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML Custom Model Demo — Full Setup Script
Reads all account/region config from config.py.

Run with: python scripts/setup_cleanrooms.py  (from the project root folder)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3, json, time

session = boto3.Session(region_name=AWS_REGION)
iam = session.client("iam")
glue = session.client("glue")
cr = session.client("cleanrooms")
crml = session.client("cleanroomsml")
sts = session.client("sts")

def log(msg):
    print(f"  → {msg}")

CLEANROOMS_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "cleanrooms.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": AWS_ACCOUNT_ID}},
    }],
}
CLEANROOMS_ML_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "cleanrooms-ml.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": AWS_ACCOUNT_ID}},
    }],
}


def create_role(role_name, trust_policy, policy_doc, description):
    try:
        resp = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_policy), Description=description)
        arn = resp["Role"]["Arn"]
        log(f"Created role: {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{role_name}"
        log(f"Role already exists: {role_name}")
    try:
        iam.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}-policy", PolicyDocument=json.dumps(policy_doc))
    except Exception as e:
        log(f"ERROR attaching inline policy to {role_name}: {e}")
        raise
    return arn


# ═══ 1. GLUE DATA CATALOG ═══
def setup_glue():
    print("\n[1/8] Setting up Glue Data Catalog...")
    try:
        glue.create_database(DatabaseInput={"Name": GLUE_DB, "Description": "AWS Clean Rooms ML demo"})
        log(f"Created database: {GLUE_DB}")
    except glue.exceptions.AlreadyExistsException:
        log(f"Database already exists: {GLUE_DB}")

    tables = {
        ADVERTISER_TABLE: {
            "columns": [
                {"Name": "user_id", "Type": "string"}, {"Name": "ad_campaign_id", "Type": "string"},
                {"Name": "impressions", "Type": "int"}, {"Name": "clicks", "Type": "int"},
                {"Name": "time_spent_seconds", "Type": "double"}, {"Name": "device_type", "Type": "string"},
                {"Name": "event_date", "Type": "string"},
            ],
            "location": f"s3://{BUCKET}/advertiser/",
        },
        RETAILER_TABLE: {
            "columns": [
                {"Name": "user_id", "Type": "string"}, {"Name": "product_category", "Type": "string"},
                {"Name": "purchase_amount", "Type": "double"}, {"Name": "purchase_count", "Type": "int"},
                {"Name": "site_visits", "Type": "int"}, {"Name": "days_since_last_purchase", "Type": "int"},
                {"Name": "last_purchase_date", "Type": "string"}, {"Name": "converted", "Type": "int"},
            ],
            "location": f"s3://{BUCKET}/retailer/",
        },
    }
    for tbl_name, tbl_cfg in tables.items():
        try:
            glue.create_table(
                DatabaseName=GLUE_DB,
                TableInput={
                    "Name": tbl_name,
                    "StorageDescriptor": {
                        "Columns": tbl_cfg["columns"],
                        "Location": tbl_cfg["location"],
                        "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                        "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                        "SerdeInfo": {
                            "SerializationLibrary": "org.apache.hadoop.hive.serde2.OpenCSVSerde",
                            "Parameters": {"separatorChar": ",", "quoteChar": '"', "escapeChar": "\\"},
                        },
                    },
                    "TableType": "EXTERNAL_TABLE",
                    "Parameters": {"classification": "csv", "skip.header.line.count": "1"},
                },
            )
            log(f"Created table: {tbl_name}")
        except glue.exceptions.AlreadyExistsException:
            log(f"Table already exists: {tbl_name}")

    # Grant Lake Formation permissions to data provider role.
    # Required in accounts where Lake Formation is active (e.g., DataZone enabled).
    # In pure-IAM accounts these grants succeed harmlessly or fail silently.
    lf = session.client("lakeformation")
    role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{ROLE_DATA_PROVIDER}"
    try:
        lf.grant_permissions(
            Principal={"DataLakePrincipalIdentifier": role_arn},
            Resource={"Database": {"Name": GLUE_DB}},
            Permissions=["DESCRIBE"],
        )
        log(f"Granted Lake Formation DESCRIBE on database {GLUE_DB}")
    except Exception as e:
        if "already" in str(e).lower():
            log("Lake Formation database permission already exists")
        else:
            log(f"Lake Formation database grant (non-fatal): {e}")

    for tbl_name in [ADVERTISER_TABLE, RETAILER_TABLE]:
        try:
            lf.grant_permissions(
                Principal={"DataLakePrincipalIdentifier": role_arn},
                Resource={"Table": {"DatabaseName": GLUE_DB, "Name": tbl_name}},
                Permissions=["SELECT", "DESCRIBE"],
            )
            log(f"Granted Lake Formation SELECT+DESCRIBE on {tbl_name}")
        except Exception as e:
            if "already" in str(e).lower():
                log(f"Lake Formation permission already exists for {tbl_name}")
            else:
                log(f"Lake Formation table grant (non-fatal): {e}")


# ═══ 2. IAM ROLES ═══
def setup_iam_roles():
    print("\n[2/8] Setting up IAM roles...")

    data_provider_arn = create_role(ROLE_DATA_PROVIDER, CLEANROOMS_TRUST, {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "GlueCatalogReadAccess",
                "Effect": "Allow",
                # Only singular Get* actions are needed; plural GetDatabases/GetTables removed
                "Action": ["glue:GetDatabase", "glue:GetTable",
                           "glue:GetPartition", "glue:GetPartitions", "glue:BatchGetPartition"],
                "Resource": [f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:catalog",
                             f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:database/{GLUE_DB}",
                             f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{GLUE_DB}/{ADVERTISER_TABLE}",
                             f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{GLUE_DB}/{RETAILER_TABLE}"],
            },
            {
                "Sid": "S3ReadDataPrefixes",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{BUCKET}/advertiser/*",
                             f"arn:aws:s3:::{BUCKET}/retailer/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "true"}},
            },
            {
                "Sid": "S3BucketAccess",
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{BUCKET}"],
                "Condition": {"Bool": {"aws:SecureTransport": "true"}},
            },
        ],
    }, "Allows AWS Clean Rooms to read Glue catalog and S3 data")

    model_provider_arn = create_role(ROLE_MODEL_PROVIDER, CLEANROOMS_ML_TRUST, {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECRPullImages",
                "Effect": "Allow",
                "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
                "Resource": [f"arn:aws:ecr:{AWS_REGION}:{AWS_ACCOUNT_ID}:repository/cleanrooms-ml-demo-training",
                             f"arn:aws:ecr:{AWS_REGION}:{AWS_ACCOUNT_ID}:repository/cleanrooms-ml-demo-inference"],
            },
            {
                "Sid": "ECRAuthToken",
                "Effect": "Allow",
                # ecr:GetAuthorizationToken does not support resource-level permissions;
                # Resource: * is required by the API.
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
        ],
    }, "Allows AWS Clean Rooms ML to pull ECR container images")

    ml_config_arn = create_role(ROLE_ML_CONFIG, CLEANROOMS_ML_TRUST, {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3DataAccess",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{BUCKET}/advertiser/*",
                             f"arn:aws:s3:::{BUCKET}/retailer/*",
                             f"arn:aws:s3:::{BUCKET}/data/*",
                             f"arn:aws:s3:::{OUTPUT_BUCKET}/cleanrooms-ml-output/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "true"}},
            },
            {
                "Sid": "S3BucketAccess",
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{OUTPUT_BUCKET}"],
                "Condition": {"Bool": {"aws:SecureTransport": "true"}},
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": f"arn:aws:logs:{AWS_REGION}:{AWS_ACCOUNT_ID}:log-group:/aws/cleanrooms*",
            },
            {
                "Sid": "CloudWatchMetrics",
                "Effect": "Allow",
                # cloudwatch:PutMetricData does not support resource-level permissions;
                # Resource: * is required by the API.
                "Action": ["cloudwatch:PutMetricData"],
                "Resource": "*",
            },
        ],
    }, "Allows AWS Clean Rooms ML to write metrics, logs, and S3 output")

    query_runner_arn = create_role(ROLE_QUERY_RUNNER, CLEANROOMS_ML_TRUST, {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CleanRoomsQueryAccess",
                "Effect": "Allow",
                # AWS Clean Rooms query actions require membership/collaboration ARNs.
                # These are scoped to this account; cross-account access is not granted.
                "Action": ["cleanrooms:StartProtectedQuery", "cleanrooms:GetProtectedQuery",
                           "cleanrooms:GetCollaboration", "cleanrooms:GetSchema",
                           "cleanrooms:GetSchemaAnalysisRule", "cleanrooms:ListSchemas"],
                "Resource": [
                    f"arn:aws:cleanrooms:{AWS_REGION}:{AWS_ACCOUNT_ID}:membership/*",
                    f"arn:aws:cleanrooms:{AWS_REGION}:{AWS_ACCOUNT_ID}:collaboration/*",
                ],
            },
        ],
    }, "Allows AWS Clean Rooms ML to run queries for ML input channels")

    log("Waiting 10s for IAM role propagation...")
    time.sleep(10)
    return {"data_provider": data_provider_arn, "model_provider": model_provider_arn,
            "ml_config": ml_config_arn, "query_runner": query_runner_arn}


# ═══ 3. AWS CLEAN ROOMS COLLABORATION ═══
def setup_collaboration():
    print("\n[3/8] Setting up AWS Clean Rooms collaboration...")
    existing = cr.list_collaborations(memberStatus="ACTIVE")
    for collab in existing.get("collaborationList", []):
        if collab["name"] == f"{PREFIX}-collaboration":
            collab_id = collab["id"]
            log(f"Collaboration already exists: {collab_id}")
            memberships = cr.list_memberships(status="ACTIVE")
            for m in memberships.get("membershipSummaries", []):
                if m["collaborationId"] == collab_id:
                    return collab_id, m["id"]
            mem_resp = cr.create_membership(
                collaborationIdentifier=collab_id, queryLogStatus="ENABLED",
                paymentConfiguration={"queryCompute": {"isResponsible": True},
                    "machineLearning": {"modelTraining": {"isResponsible": True}, "modelInference": {"isResponsible": True}}})
            return collab_id, mem_resp["membership"]["id"]

    resp = cr.create_collaboration(
        name=f"{PREFIX}-collaboration",
        description="Customer Propensity Scoring — Advertiser + Retailer collaboration",
        creatorDisplayName="Advertiser",
        creatorMemberAbilities=["CAN_QUERY", "CAN_RECEIVE_RESULTS"],
        creatorMLMemberAbilities={"customMLMemberAbilities": ["CAN_RECEIVE_MODEL_OUTPUT", "CAN_RECEIVE_INFERENCE_OUTPUT"]},
        creatorPaymentConfiguration={"queryCompute": {"isResponsible": True},
            "machineLearning": {"modelTraining": {"isResponsible": True}, "modelInference": {"isResponsible": True}}},
        members=[], queryLogStatus="ENABLED",
        dataEncryptionMetadata={"allowCleartext": True, "allowDuplicates": True,
            "allowJoinsOnColumnsWithDifferentNames": True, "preserveNulls": False},
    )
    collab_id = resp["collaboration"]["id"]
    log(f"Created collaboration: {collab_id}")

    mem_resp = cr.create_membership(
        collaborationIdentifier=collab_id, queryLogStatus="ENABLED",
        paymentConfiguration={"queryCompute": {"isResponsible": True},
            "machineLearning": {"modelTraining": {"isResponsible": True}, "modelInference": {"isResponsible": True}}})
    membership_id = mem_resp["membership"]["id"]
    log(f"Created membership: {membership_id}")
    return collab_id, membership_id


# ═══ 4. CONFIGURED TABLES + ASSOCIATIONS ═══
def setup_configured_tables(membership_id, roles):
    print("\n[4/8] Setting up configured tables and associations...")
    columns_by_table = {
        "advertiser": ["user_id", "ad_campaign_id", "impressions", "clicks", "time_spent_seconds", "device_type", "event_date"],
        "retailer": ["user_id", "product_category", "purchase_amount", "purchase_count", "site_visits", "days_since_last_purchase", "last_purchase_date", "converted"],
    }
    table_ids = {}
    for table_name, glue_table in [("advertiser", ADVERTISER_TABLE), ("retailer", RETAILER_TABLE)]:
        ct_name = f"{PREFIX}-{table_name}"
        existing = cr.list_configured_tables()
        ct_arn = ct_id = None
        for ct in existing.get("configuredTableSummaries", []):
            if ct["name"] == ct_name:
                ct_arn, ct_id = ct["arn"], ct["id"]
                log(f"Configured table already exists: {ct_name} ({ct_id})")
                break
        if not ct_arn:
            resp = cr.create_configured_table(
                name=ct_name, description=f"{table_name.title()} data for propensity scoring",
                tableReference={"glue": {"tableName": glue_table, "databaseName": GLUE_DB}},
                allowedColumns=columns_by_table[table_name], analysisMethod="DIRECT_QUERY")
            ct_arn, ct_id = resp["configuredTable"]["arn"], resp["configuredTable"]["id"]
            log(f"Created configured table: {ct_name} ({ct_id})")

        try:
            cr.create_configured_table_analysis_rule(
                configuredTableIdentifier=ct_id, analysisRuleType="LIST",
                analysisRulePolicy={"v1": {"list": {
                    "joinColumns": ["user_id"],
                    "listColumns": [col["Name"] for col in glue.get_table(DatabaseName=GLUE_DB, Name=glue_table)["Table"]["StorageDescriptor"]["Columns"] if col["Name"] != "user_id"],
                    "allowedJoinOperators": ["OR"],
                    "additionalAnalyses": "ALLOWED",
                }}})
            log(f"Created LIST analysis rule for {ct_name}")
        except (cr.exceptions.ConflictException, cr.exceptions.ClientError) as e:
            if "already has" in str(e) or "Conflict" in str(e):
                log(f"Analysis rule already exists for {ct_name}")
            else:
                raise

        assoc_name = f"{table_name}_association"
        try:
            cr.create_configured_table_association(
                membershipIdentifier=membership_id, configuredTableIdentifier=ct_id,
                name=assoc_name, description=f"{table_name.title()} data association",
                roleArn=roles["data_provider"])
            log(f"Created table association: {assoc_name}")
        except cr.exceptions.ConflictException:
            log(f"Table association already exists: {assoc_name}")
        table_ids[table_name] = ct_id
    return table_ids


# ═══ 5. ML CONFIGURATION ═══
def setup_ml_configuration(membership_id, roles):
    print("\n[5/8] Setting up ML configuration...")
    try:
        crml.put_ml_configuration(
            membershipIdentifier=membership_id,
            defaultOutputLocation={
                "destination": {"s3Destination": {"s3Uri": f"s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/"}},
                "roleArn": roles["ml_config"],
            })
        log("ML configuration set")
    except Exception as e:
        log(f"ML configuration: {e}")


# ═══ 6. CONFIGURED MODEL ALGORITHM ═══
def setup_model_algorithm(roles):
    print("\n[6/8] Setting up configured model algorithm...")
    algo_name = f"{PREFIX}-propensity-model"
    existing = crml.list_configured_model_algorithms()
    for algo in existing.get("configuredModelAlgorithms", []):
        if algo["name"] == algo_name:
            log(f"Model algorithm already exists: {algo['configuredModelAlgorithmArn']}")
            return algo["configuredModelAlgorithmArn"]

    resp = crml.create_configured_model_algorithm(
        name=algo_name, description="Customer propensity scoring model",
        roleArn=roles["model_provider"],
        trainingContainerConfig={
            "imageUri": TRAINING_IMAGE,
            "entrypoint": ["python", "/opt/ml/code/train.py"],
            "metricDefinitions": [
                {"name": "accuracy", "regex": r'"accuracy": ([0-9.]+)'},
                {"name": "roc_auc", "regex": r'"roc_auc": ([0-9.]+)'},
                {"name": "f1", "regex": r'"f1": ([0-9.]+)'},
            ],
        },
        inferenceContainerConfig={"imageUri": INFERENCE_IMAGE})
    algo_arn = resp["configuredModelAlgorithmArn"]
    log(f"Created model algorithm: {algo_arn}")
    return algo_arn


# ═══ 7. MODEL ALGORITHM ASSOCIATION ═══
def setup_model_algorithm_association(membership_id, algo_arn, collab_id):
    print("\n[7/8] Associating model algorithm to collaboration...")
    assoc_name = f"{PREFIX}-propensity-assoc"
    existing = crml.list_configured_model_algorithm_associations(membershipIdentifier=membership_id)
    for assoc in existing.get("configuredModelAlgorithmAssociations", []):
        if assoc.get("name") == assoc_name:
            log(f"Association already exists: {assoc['configuredModelAlgorithmAssociationArn']}")
            return assoc["configuredModelAlgorithmAssociationArn"]

    resp = crml.create_configured_model_algorithm_association(
        membershipIdentifier=membership_id, configuredModelAlgorithmArn=algo_arn,
        name=assoc_name, description="Propensity model algorithm association")
    assoc_arn = resp["configuredModelAlgorithmAssociationArn"]
    log(f"Created algorithm association: {assoc_arn}")
    return assoc_arn


# ═══ MAIN ═══
def setup_association_analysis_rules(membership_id, algo_assoc_arn):
    """Create collaboration-level analysis rules on each configured table association."""
    print("\n[8/8] Setting up association-level analysis rules...")
    assocs = cr.list_configured_table_associations(membershipIdentifier=membership_id)["configuredTableAssociationSummaries"]
    for a in assocs:
        name, aid = a["name"], a["id"]
        try:
            cr.create_configured_table_association_analysis_rule(
                membershipIdentifier=membership_id,
                configuredTableAssociationIdentifier=aid,
                analysisRuleType="LIST",
                analysisRulePolicy={"v1": {"list": {
                    "allowedResultReceivers": [AWS_ACCOUNT_ID],
                    "allowedAdditionalAnalyses": [algo_assoc_arn],
                }}})
            log(f"Created association analysis rule for {name}")
        except (cr.exceptions.ConflictException, cr.exceptions.ClientError) as e:
            if "already" in str(e).lower() or "conflict" in str(e).lower():
                log(f"Association analysis rule already exists for {name}")
            else:
                raise


def main():
    print("=" * 60)
    print("AWS Clean Rooms ML Custom Model Demo — Setup")
    print("=" * 60)
    print(f"Account:  {AWS_ACCOUNT_ID}")
    print(f"Region:   {AWS_REGION}")
    print(f"Bucket:   {BUCKET}")

    try:
        identity = sts.get_caller_identity()
        assert identity["Account"] == AWS_ACCOUNT_ID
        log(f"Authenticated as: {identity['Arn']}")
    except Exception as e:
        print(f"ERROR: AWS credentials not valid: {e}")
        sys.exit(1)

    setup_glue()
    roles = setup_iam_roles()
    collab_id, membership_id = setup_collaboration()
    if not membership_id:
        print("ERROR: Could not find membership ID.")
        sys.exit(1)

    setup_configured_tables(membership_id, roles)
    setup_ml_configuration(membership_id, roles)
    algo_arn = setup_model_algorithm(roles)
    assoc_arn = setup_model_algorithm_association(membership_id, algo_arn, collab_id)
    setup_association_analysis_rules(membership_id, assoc_arn)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nCollaboration ID:  {collab_id}")
    print(f"Membership ID:     {membership_id}")
    print(f"Algorithm ARN:     {algo_arn}")
    print(f"Association ARN:   {assoc_arn}")
    print(f"\nConsole: https://{AWS_REGION}.console.aws.amazon.com/cleanrooms/home?region={AWS_REGION}#/collaborations/{collab_id}")
    print(f"\nNext: python scripts/run_cleanrooms_ml.py")


if __name__ == "__main__":
    main()
