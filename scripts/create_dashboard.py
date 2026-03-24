# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML — Create QuickSight Dashboard (Step 6)
Reads all account/region config from config.py.

Run with: python scripts/create_dashboard.py  (from the project root folder)

Prerequisites:
  - run_cleanrooms_ml.py must have completed successfully
  - Inference output must exist at s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/
  - metrics.json must exist in the trained model output in S3

What this script does (idempotent — safe to re-run):
  1. Register QuickSight account (skip if already exists)
  2. Register QuickSight admin user (skip if already exists)
  3. Create Glue tables for inference output + model metrics
  4. Create Athena data source in QuickSight
  5. Create QuickSight SPICE datasets
  6. Create analysis + publish dashboard (5 sheets)
  7. Print dashboard URL
"""

import sys, os, json, time, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate(require_qs_email=True)

import boto3
from botocore.exceptions import ClientError

# ─── Clients ──────────────────────────────────────────────
# QuickSight user/account management APIs are global and MUST use us-east-1.
# All other QuickSight APIs (datasets, dashboards, analyses) use AWS_REGION.
session     = boto3.Session(region_name=AWS_REGION)
session_iam = boto3.Session(region_name="us-east-1")   # QS identity plane
qs          = session.client("quicksight")              # data plane (regional)
qs_iam      = session_iam.client("quicksight")          # identity plane (us-east-1)
glue        = session.client("glue")
s3          = session.client("s3")
iam         = session.client("iam")
sts         = session.client("sts")

# ─── Resource IDs (stable, derived from PREFIX) ───────────
QS_DATASOURCE_ID  = f"{PREFIX}-athena-source"
QS_DS_INFERENCE   = f"{PREFIX}-ds-inference"
QS_ANALYSIS_ID    = f"{PREFIX}-propensity-analysis"
QS_DASHBOARD_ID   = f"{PREFIX}-propensity-dashboard"

# Glue table created by this script (source tables already exist from setup_cleanrooms.py)
INFERENCE_TABLE   = "inference_output"


def log(msg):
    print(f"  → {msg}")


# ═══════════════════════════════════════════════════════════
# SECTION 1 — QuickSight account registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_account():
    """Register QuickSight ENTERPRISE account. No-op if already registered."""
    print("\n[1/6] Ensuring QuickSight account...")

    # Check if already subscribed — account APIs must use us-east-1 endpoint
    try:
        resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
        status = resp["AccountInfo"]["AccountSubscriptionStatus"]
        log(f"QuickSight account already exists (status: {status})")
        if status not in ("ACCOUNT_CREATED", "ACTIVE"):
            print(f"  WARNING: QuickSight account status is '{status}'. "
                  "It may still be provisioning — waiting up to 60s...")
            _wait_for_qs_account()
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceNotFoundException", "AccessDeniedException"):
            raise

    # Not yet registered — create it
    log(f"Registering QuickSight ENTERPRISE account (notification email: {QS_NOTIFICATION_EMAIL})")
    try:
        qs_iam.create_account_subscription(
            AwsAccountId=AWS_ACCOUNT_ID,
            AccountName=f"{PREFIX}-{AWS_ACCOUNT_ID}",
            Edition="ENTERPRISE",
            AuthenticationMethod="IAM_AND_QUICKSIGHT",
            NotificationEmail=QS_NOTIFICATION_EMAIL,
        )
        log("QuickSight account registration submitted — waiting for activation...")
        _wait_for_qs_account()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("ResourceExistsException", "ConflictException"):
            log("QuickSight account already exists (race condition) — continuing")
        else:
            raise


def _wait_for_qs_account(max_wait=120):
    """Poll until QuickSight account status is ACCOUNT_CREATED."""
    for _ in range(max_wait // 10):
        try:
            resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
            status = resp["AccountInfo"]["AccountSubscriptionStatus"]
            if status == "ACCOUNT_CREATED":
                log("QuickSight account is ACTIVE")
                return
            log(f"  Account status: {status} — waiting...")
        except ClientError:
            pass
        time.sleep(10)
    print("  WARNING: QuickSight account did not reach ACCOUNT_CREATED within timeout. Continuing anyway.")


# ═══════════════════════════════════════════════════════════
# SECTION 2 — QuickSight admin user registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_user():
    """Register the current IAM caller as a QuickSight ADMIN user. No-op if exists."""
    print("\n[2/6] Ensuring QuickSight admin user...")

    identity = sts.get_caller_identity()
    caller_arn = identity["Arn"]

    # Derive username: for assumed roles QS uses "RoleName/SessionName"
    arn_parts = caller_arn.split(":")
    raw_name  = arn_parts[-1]
    if raw_name.startswith("assumed-role/"):
        # assumed-role/RoleName/SessionName → RoleName/SessionName
        username = "/".join(raw_name.split("/")[1:])
    else:
        username = raw_name.split("/")[-1]

    try:
        qs_iam.describe_user(
            AwsAccountId=AWS_ACCOUNT_ID,
            Namespace="default",
            UserName=username,
        )
        log(f"QuickSight user already exists: {username}")
        return username
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log(f"Registering QuickSight ADMIN user: {username}")
    # For assumed-role ARNs, IamArn must be the role ARN (without session),
    # and SessionName must be the session name only (last segment after final '/')
    is_assumed_role = "assumed-role" in caller_arn
    if is_assumed_role:
        # arn:aws:sts::123:assumed-role/RoleName/SessionName
        # → role_arn = arn:aws:iam::123:role/RoleName
        parts = caller_arn.split(":")
        role_path = parts[-1]  # assumed-role/RoleName/SessionName
        role_parts = role_path.split("/")
        role_name = role_parts[1]
        session_name = role_parts[2]
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{role_name}"
    else:
        role_arn = caller_arn
        session_name = None

    qs_iam.register_user(
        AwsAccountId=AWS_ACCOUNT_ID,
        Namespace="default",
        IdentityType="IAM",
        IamArn=role_arn,
        UserRole="ADMIN",
        Email=QS_NOTIFICATION_EMAIL,
        **({"SessionName": session_name} if session_name else {}),
    )
    log(f"Registered user: {username}")
    return username


def _qs_user_arn(username):
    # User ARNs always reference us-east-1 (identity plane is global/us-east-1)
    return f"arn:aws:quicksight:us-east-1:{AWS_ACCOUNT_ID}:user/default/{username}"


# ═══════════════════════════════════════════════════════════
# SECTION 3 — Prepare dashboard data in S3 + Glue
# ═══════════════════════════════════════════════════════════

def prepare_glue_tables():
    """Register the inference output CSV as a Glue table (Athena-queryable)."""
    print("\n[3/6] Preparing Glue tables for dashboard data...")
    _ensure_glue_db()
    _register_inference_table()


def _ensure_glue_db():
    try:
        glue.create_database(DatabaseInput={"Name": GLUE_DB, "Description": "AWS Clean Rooms ML demo"})
        log(f"Created Glue database: {GLUE_DB}")
    except glue.exceptions.AlreadyExistsException:
        log(f"Glue database already exists: {GLUE_DB}")


def _register_inference_table():
    """Register inference output CSV location as a Glue external table."""
    columns = [
        {"Name": "propensity_score",    "Type": "double"},
        {"Name": "predicted_converter", "Type": "int"},
        {"Name": "ad_campaign_id",      "Type": "string"},
        {"Name": "device_type",         "Type": "string"},
        {"Name": "product_category",    "Type": "string"},
        {"Name": "purchase_amount",     "Type": "double"},
        {"Name": "impressions",         "Type": "int"},
        {"Name": "clicks",              "Type": "int"},
    ]
    table_input = {
        "Name": INFERENCE_TABLE,
        "StorageDescriptor": {
            "Columns": columns,
            "Location": f"s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/",
            "InputFormat":  "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.serde2.OpenCSVSerde",
                "Parameters": {
                    "separatorChar": ",",
                    "quoteChar": '"',
                    "skip.header.line.count": "1",
                },
            },
        },
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": "csv"},
    }
    try:
        glue.create_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Created Glue table: {INFERENCE_TABLE}")
    except glue.exceptions.AlreadyExistsException:
        glue.update_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Updated Glue table: {INFERENCE_TABLE}")




# ═══════════════════════════════════════════════════════════
# SECTION 3b — Grant QuickSight access to S3 + Athena
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_s3_access():
    """
    Grant QuickSight permission to read the output S3 bucket via the
    QuickSight managed service role. This is required for Athena queries
    that read from S3 — without it visuals show 'insufficient permissions'.
    Uses the update_ip_restriction / register_account_customization approach
    via the IAM inline policy on the QuickSight service role.
    """
    qs_service_role = f"aws-quicksight-service-role-v0"
    qs_s3_role      = f"aws-quicksight-s3-consumers-role-v0"

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CleanRoomsMLOutputBucketAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
            {
                "Sid": "AthenaAccess",
                "Effect": "Allow",
                "Action": [
                    "athena:BatchGetQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:GetQueryResultsStream",
                    "athena:ListQueryExecutions",
                    "athena:StartQueryExecution",
                    "athena:StopQueryExecution",
                    "athena:GetWorkGroup",
                ],
                "Resource": "*",
            },
            {
                "Sid": "GlueAccess",
                "Effect": "Allow",
                "Action": [
                    "glue:GetDatabase", "glue:GetDatabases",
                    "glue:GetTable", "glue:GetTables",
                    "glue:GetPartition", "glue:GetPartitions",
                    "glue:BatchGetPartition",
                ],
                "Resource": [
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:catalog",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:database/{GLUE_DB}",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{GLUE_DB}/*",
                ],
            },
            {
                "Sid": "AthenaResultsBucket",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket",
                           "s3:GetBucketLocation", "s3:AbortMultipartUpload"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
        ],
    }

    for role_name in [qs_service_role, qs_s3_role]:
        try:
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=f"cleanrooms-ml-demo-qs-access",
                PolicyDocument=json.dumps(policy_doc),
            )
            log(f"Granted S3/Athena/Glue access to QuickSight role: {role_name}")
        except iam.exceptions.NoSuchEntityException:
            log(f"QuickSight role not found (skipping): {role_name}")
        except Exception as e:
            log(f"Could not update role {role_name} (non-fatal): {e}")


def ensure_datasource(user_arn):
    """Create (or verify) the Athena data source in QuickSight."""
    print("\n[4/6] Ensuring QuickSight Athena data source...")

    try:
        qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        log(f"Data source already exists: {QS_DATASOURCE_ID}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    qs.create_data_source(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSourceId=QS_DATASOURCE_ID,
        Name="CleanRoomsML — Athena",
        Type="ATHENA",
        DataSourceParameters={"AthenaParameters": {"WorkGroup": "primary"}},
        Permissions=[{
            "Principal": user_arn,
            "Actions": [
                "quicksight:DescribeDataSource",
                "quicksight:DescribeDataSourcePermissions",
                "quicksight:PassDataSource",
                "quicksight:UpdateDataSource",
                "quicksight:DeleteDataSource",
                "quicksight:UpdateDataSourcePermissions",
            ],
        }],
    )
    log(f"Created Athena data source: {QS_DATASOURCE_ID}")
    # Wait for CREATION_SUCCESSFUL
    _wait_for_datasource()


def _wait_for_datasource(max_wait=60):
    for _ in range(max_wait // 5):
        resp = qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        status = resp["DataSource"]["Status"]
        if status == "CREATION_SUCCESSFUL":
            log("Data source is ready")
            return
        if "FAILED" in status:
            raise RuntimeError(f"Data source creation failed: {status} — "
                               f"{resp['DataSource'].get('DataSourceErrorInfo', {})}")
        time.sleep(5)
    print("  WARNING: Data source did not reach CREATION_SUCCESSFUL within timeout.")


def _athena_physical_table(table_name, sql):
    """Helper: build a PhysicalTableMap entry using a CustomSql source."""
    return {
        "CustomSql": {
            "DataSourceArn": f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:datasource/{QS_DATASOURCE_ID}",
            "Name": table_name,
            "SqlQuery": sql,
            "Columns": [],  # QuickSight infers columns from the query
        }
    }


def _dataset_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDataSet",
            "quicksight:DescribeDataSetPermissions",
            "quicksight:PassDataSet",
            "quicksight:DescribeIngestion",
            "quicksight:ListIngestions",
            "quicksight:UpdateDataSet",
            "quicksight:DeleteDataSet",
            "quicksight:CreateIngestion",
            "quicksight:CancelIngestion",
            "quicksight:UpdateDataSetPermissions",
        ],
    }]


def ensure_datasets(user_arn):
    """Create (or update) the inference SPICE dataset."""
    print("\n[5/6] Ensuring QuickSight datasets...")

    datasource_arn = (
        f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:datasource/{QS_DATASOURCE_ID}"
    )

    datasets = [
        {
            "DataSetId": QS_DS_INFERENCE,
            "Name": "Propensity Inference Output",
            "sql": (
                f"SELECT propensity_score, predicted_converter, "
                f"ad_campaign_id, device_type, product_category, "
                f"purchase_amount, impressions, clicks, "
                f"CASE WHEN propensity_score > 0.7 THEN 'High' "
                f"     WHEN propensity_score >= 0.3 THEN 'Medium' "
                f"     ELSE 'Low' END AS propensity_segment, "
                f"CAST(CEIL(PERCENT_RANK() OVER (ORDER BY propensity_score) * 10) AS INTEGER) AS score_decile, "
                f"purchase_amount * propensity_score AS revenue_impact "
                f"FROM {GLUE_DB}.{INFERENCE_TABLE}"
            ),
        },
    ]

    for ds in datasets:
        _upsert_dataset(ds, datasource_arn, user_arn)


def _upsert_dataset(ds_cfg, datasource_arn, user_arn):
    ds_id   = ds_cfg["DataSetId"]
    ds_name = ds_cfg["Name"]
    sql     = ds_cfg["sql"]

    physical_id = f"{ds_id}-physical"
    logical_id  = f"{ds_id}-logical"

    physical_table_map = {
        physical_id: {
            "CustomSql": {
                "DataSourceArn": datasource_arn,
                "Name": ds_name,
                "SqlQuery": sql,
                "Columns": _infer_columns_for(ds_id),
            }
        }
    }
    logical_table_map = {
        logical_id: {
            "Alias": ds_name,
            "Source": {"PhysicalTableId": physical_id},
        }
    }

    kwargs = dict(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSetId=ds_id,
        Name=ds_name,
        PhysicalTableMap=physical_table_map,
        LogicalTableMap=logical_table_map,
        ImportMode="DIRECT_QUERY",
        Permissions=_dataset_permissions(user_arn),
    )

    try:
        qs.describe_data_set(AwsAccountId=AWS_ACCOUNT_ID, DataSetId=ds_id)
        # Exists — update (permissions not part of update_data_set)
        update_kwargs = {k: v for k, v in kwargs.items() if k != "Permissions"}
        qs.update_data_set(**update_kwargs)
        log(f"Created/updated dataset: {ds_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_data_set(**kwargs)
        log(f"Created/updated dataset: {ds_name}")


def _infer_columns_for(ds_id):
    """Return explicit column definitions for the inference dataset's CustomSql."""
    if ds_id == QS_DS_INFERENCE:
        return [
            {"Name": "propensity_score",    "Type": "DECIMAL"},
            {"Name": "predicted_converter", "Type": "INTEGER"},
            {"Name": "ad_campaign_id",      "Type": "STRING"},
            {"Name": "device_type",         "Type": "STRING"},
            {"Name": "product_category",    "Type": "STRING"},
            {"Name": "purchase_amount",     "Type": "DECIMAL"},
            {"Name": "impressions",         "Type": "INTEGER"},
            {"Name": "clicks",              "Type": "INTEGER"},
            {"Name": "propensity_segment",  "Type": "STRING"},
            {"Name": "score_decile",        "Type": "INTEGER"},
            {"Name": "revenue_impact",      "Type": "DECIMAL"},
        ]
    return []


# ═══════════════════════════════════════════════════════════
# SECTION 5 — Dashboard definition helpers
# ═══════════════════════════════════════════════════════════

# Alias used in DataSetIdentifierDeclarations
_DS_INFERENCE = "inference"


def _dataset_declarations():
    return [
        {"Identifier": _DS_INFERENCE, "DataSetArn": f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:dataset/{QS_DS_INFERENCE}"},
    ]


def _col(ds_alias, col_name):
    return {"DataSetIdentifier": ds_alias, "ColumnName": col_name}


def _num_measure(field_id, ds_alias, col_name, agg="AVERAGE"):
    return {"NumericalMeasureField": {
        "FieldId": field_id,
        "Column": _col(ds_alias, col_name),
        "AggregationFunction": {"SimpleNumericalAggregation": agg},
    }}


def _num_dim(field_id, ds_alias, col_name):
    return {"NumericalDimensionField": {"FieldId": field_id, "Column": _col(ds_alias, col_name)}}


def _cat_dim(field_id, ds_alias, col_name):
    return {"CategoricalDimensionField": {"FieldId": field_id, "Column": _col(ds_alias, col_name)}}


def _visual_title(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


def _visual_subtitle(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


# ── Sheet 1: Model Performance Overview ──────────────────

# ── Sheet 2: Propensity Score Distribution ────────────────

def _sheet2():
    # Histogram approximation: bar chart with score_decile on X, count on Y
    histogram = {"BarChartVisual": {
        "VisualId": "bar-score-dist",
        "Title": _visual_title("Score Distribution by Decile"),
        "Subtitle": _visual_subtitle("Each bar shows how many records fall in each propensity score decile (1=lowest, 10=highest). A right-skewed distribution means most users have high propensity — ideal for targeting."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("decile-dim", _DS_INFERENCE, "score_decile")],
                    "Values":   [_num_measure("decile-cnt", _DS_INFERENCE, "propensity_score", "COUNT")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "decile-dim", "Direction": "ASC"}}],
            },
        },
    }}

    # Donut: segment distribution
    donut = {"PieChartVisual": {
        "VisualId": "donut-segments",
        "Title": _visual_title("Users by Propensity Segment"),
        "Subtitle": _visual_subtitle("Breakdown of all scored records into High (>0.7), Medium (0.3–0.7), and Low (<0.3) propensity segments. High-segment users are the primary targeting audience."),
        "ChartConfiguration": {
            "FieldWells": {
                "PieChartAggregatedFieldWells": {
                    "Category": [_cat_dim("seg-dim", _DS_INFERENCE, "propensity_segment")],
                    "Values":   [_num_measure("seg-cnt", _DS_INFERENCE, "propensity_score", "COUNT")],
                }
            },
            "DonutOptions": {"ArcOptions": {"ArcThickness": "MEDIUM"}},
        },
    }}

    # Table: decile lift table
    decile_table = {"TableVisual": {
        "VisualId": "tbl-decile-lift",
        "Title": _visual_title("Propensity Decile Lift Table"),
        "Subtitle": _visual_subtitle("For each score decile, shows record count, average propensity score, and conversion rate. Higher deciles should show higher conversion rates — this is the model's lift over random targeting."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_num_dim("lift-decile", _DS_INFERENCE, "score_decile")],
                    "Values":  [
                        _num_measure("lift-cnt",   _DS_INFERENCE, "propensity_score", "COUNT"),
                        _num_measure("lift-score", _DS_INFERENCE, "propensity_score", "AVERAGE"),
                        _num_measure("lift-conv",  _DS_INFERENCE, "predicted_converter", "AVERAGE"),
                    ],
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "lift-decile", "Direction": "ASC"}}],
            },
        },
    }}

    # Bar: converters vs non-converters avg score
    conv_bar = {"BarChartVisual": {
        "VisualId": "bar-conv-vs-nonconv",
        "Title": _visual_title("Avg Propensity: Converters vs Non-Converters"),
        "Subtitle": _visual_subtitle("Validates model quality: actual converters (label=1) should have a higher average propensity score than non-converters (label=0). A clear gap confirms the model is discriminating correctly."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("conv-dim", _DS_INFERENCE, "predicted_converter")],
                    "Values":   [_num_measure("conv-score", _DS_INFERENCE, "propensity_score", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    return {
        "SheetId": "sheet-2",
        "Name": "Score Distribution",
        "Visuals": [histogram, donut, decile_table, conv_bar],
    }


# ── Sheet 3: Campaign & Channel Analysis ─────────────────

def _sheet3():
    def _avg_score_bar(vid, title, subtitle, dim_field_id, dim_col):
        return {"BarChartVisual": {
            "VisualId": vid,
            "Title": _visual_title(title),
            "Subtitle": _visual_subtitle(subtitle),
            "ChartConfiguration": {
                "FieldWells": {
                    "BarChartAggregatedFieldWells": {
                        "Category": [_cat_dim(dim_field_id, _DS_INFERENCE, dim_col)],
                        "Values":   [_num_measure(f"{vid}-val", _DS_INFERENCE, "propensity_score", "AVERAGE")],
                    }
                },
                "Orientation": "HORIZONTAL",
                "SortConfiguration": {
                    "CategorySort": [{"FieldSort": {"FieldId": f"{vid}-val", "Direction": "DESC"}}],
                },
            },
        }}

    campaign_bar = _avg_score_bar("bar-campaign",
        "Avg Propensity by Campaign",
        "Which ad campaigns attract the highest-propensity users. Campaigns with higher scores are driving more purchase-ready audiences — prioritise budget toward these.",
        "camp-dim", "ad_campaign_id")
    device_bar   = _avg_score_bar("bar-device",
        "Avg Propensity by Device Type",
        "Average propensity score by the device used during ad engagement. Higher scores on a device type suggest that channel reaches more purchase-ready users.",
        "dev-dim", "device_type")
    category_bar = _avg_score_bar("bar-category",
        "Avg Propensity by Product Category",
        "Which product categories are associated with the highest purchase intent. Use this to align ad creative and inventory focus with high-propensity categories.",
        "cat-dim", "product_category")

    # Scatter: impressions vs propensity_score, colored by predicted_converter
    scatter = {"ScatterPlotVisual": {
        "VisualId": "scatter-impressions-score",
        "Title": _visual_title("Impressions vs Propensity Score"),
        "Subtitle": _visual_subtitle("Each point is a campaign-user combination. Users with more ad impressions tend to have higher propensity scores (retargeting effect). Points coloured by conversion status show whether high-impression users actually converted."),
        "ChartConfiguration": {
            "FieldWells": {
                "ScatterPlotCategoricallyAggregatedFieldWells": {
                    "XAxis":  [_num_measure("sc-x", _DS_INFERENCE, "impressions", "AVERAGE")],
                    "YAxis":  [_num_measure("sc-y", _DS_INFERENCE, "propensity_score", "AVERAGE")],
                    "Category": [_num_dim("sc-conv", _DS_INFERENCE, "predicted_converter")],
                    "Size":   [_num_measure("sc-sz", _DS_INFERENCE, "propensity_score", "COUNT")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-3",
        "Name": "Campaign & Channel",
        "Visuals": [campaign_bar, device_bar, category_bar, scatter],
    }


# ── Sheet 4: Propensity Segment Deep Dive ────────────────

def _sheet4():
    # Segment summary table
    segment_table = {"TableVisual": {
        "VisualId": "tbl-segment-summary",
        "Title": _visual_title("Segment Summary"),
        "Subtitle": _visual_subtitle("Aggregated view of High, Medium, and Low propensity segments. Shows record count, average score, conversion rate, average purchase amount, and average clicks per segment. Use filters above to drill into specific campaigns or categories."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_cat_dim("seg-grp", _DS_INFERENCE, "propensity_segment")],
                    "Values":  [
                        _num_measure("seg-cnt",    _DS_INFERENCE, "propensity_score",    "COUNT"),
                        _num_measure("seg-score",  _DS_INFERENCE, "propensity_score",    "AVERAGE"),
                        _num_measure("seg-conv",   _DS_INFERENCE, "predicted_converter", "AVERAGE"),
                        _num_measure("seg-amount", _DS_INFERENCE, "purchase_amount",     "AVERAGE"),
                        _num_measure("seg-clicks", _DS_INFERENCE, "clicks",              "AVERAGE"),
                    ],
                }
            },
        },
    }}

    # Top-N records table (unaggregated — shows individual rows)
    top_records = {"TableVisual": {
        "VisualId": "tbl-top-records",
        "Title": _visual_title("Top Scoring Records"),
        "Subtitle": _visual_subtitle("Individual records ranked by propensity score (highest first). Each row represents one user-campaign-category combination. Use this to identify the specific ad and product context driving the highest purchase intent."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableUnaggregatedFieldWells": {
                    "Values": [
                        {"FieldId": "tr-score",  "Column": _col(_DS_INFERENCE, "propensity_score")},
                        {"FieldId": "tr-conv",   "Column": _col(_DS_INFERENCE, "predicted_converter")},
                        {"FieldId": "tr-camp",   "Column": _col(_DS_INFERENCE, "ad_campaign_id")},
                        {"FieldId": "tr-dev",    "Column": _col(_DS_INFERENCE, "device_type")},
                        {"FieldId": "tr-cat",    "Column": _col(_DS_INFERENCE, "product_category")},
                        {"FieldId": "tr-amount", "Column": _col(_DS_INFERENCE, "purchase_amount")},
                        {"FieldId": "tr-impr",   "Column": _col(_DS_INFERENCE, "impressions")},
                        {"FieldId": "tr-clicks", "Column": _col(_DS_INFERENCE, "clicks")},
                    ]
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "tr-score", "Direction": "DESC"}}],
            },
            "PaginatedReportOptions": {"VerticalOverflowVisibility": "VISIBLE"},
        },
    }}

    # Cross-tab: segment × campaign conversion rate
    crosstab = {"PivotTableVisual": {
        "VisualId": "pivot-segment-campaign",
        "Title": _visual_title("Conversion Rate: Segment × Campaign"),
        "Subtitle": _visual_subtitle("Cross-tab of propensity segment (rows) vs ad campaign (columns), showing average conversion rate per cell. Identifies which campaign + segment combinations deliver the highest conversion density for targeting prioritisation."),
        "ChartConfiguration": {
            "FieldWells": {
                "PivotTableAggregatedFieldWells": {
                    "Rows":    [_cat_dim("pt-seg",  _DS_INFERENCE, "propensity_segment")],
                    "Columns": [_cat_dim("pt-camp", _DS_INFERENCE, "ad_campaign_id")],
                    "Values":  [_num_measure("pt-conv", _DS_INFERENCE, "predicted_converter", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-4",
        "Name": "Segment Deep Dive",
        "Visuals": [segment_table, top_records, crosstab],
    }


# ── Sheet 5: Business Impact ──────────────────────────────

def _sheet5():
    # Cumulative gains: % converters captured by top N deciles
    gains_bar = {"BarChartVisual": {
        "VisualId": "bar-cumulative-gains",
        "Title": _visual_title("Converters Captured by Propensity Decile"),
        "Subtitle": _visual_subtitle("Shows how many actual converters fall in each score decile. A good model concentrates converters in the top deciles (9–10). If you only target the top 2 deciles, this chart shows what % of all converters you would reach."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("cg-decile", _DS_INFERENCE, "score_decile")],
                    "Values":   [_num_measure("cg-conv", _DS_INFERENCE, "predicted_converter", "SUM")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "cg-decile", "Direction": "ASC"}}],
            },
        },
    }}

    # Revenue impact by segment
    revenue_bar = {"BarChartVisual": {
        "VisualId": "bar-revenue-impact",
        "Title": _visual_title("Estimated Revenue Impact by Segment"),
        "Subtitle": _visual_subtitle("Estimated revenue potential per segment, calculated as purchase_amount × propensity_score. High-segment users represent the largest revenue opportunity — use this to size the business case for targeted campaigns."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("rev-seg", _DS_INFERENCE, "propensity_segment")],
                    "Values":   [_num_measure("rev-val", _DS_INFERENCE, "revenue_impact", "SUM")],
                }
            },
            "Orientation": "HORIZONTAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "rev-val", "Direction": "DESC"}}],
            },
        },
    }}

    # Heatmap: product_category × ad_campaign_id, avg propensity
    heatmap = {"HeatMapVisual": {
        "VisualId": "heatmap-cat-campaign",
        "Title": _visual_title("Avg Propensity: Category × Campaign"),
        "Subtitle": _visual_subtitle("Heatmap showing average propensity score for each product category and ad campaign combination. Darker cells indicate higher purchase intent. Use this to identify the most effective campaign-category pairings for joint targeting."),
        "ChartConfiguration": {
            "FieldWells": {
                "HeatMapAggregatedFieldWells": {
                    "Rows":    [_cat_dim("hm-cat",  _DS_INFERENCE, "product_category")],
                    "Columns": [_cat_dim("hm-camp", _DS_INFERENCE, "ad_campaign_id")],
                    "Values":  [_num_measure("hm-val", _DS_INFERENCE, "propensity_score", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-5",
        "Name": "Business Impact",
        "Visuals": [gains_bar, revenue_bar, heatmap],
    }


# ═══════════════════════════════════════════════════════════
# SECTION 6 — Create analysis + publish dashboard
# ═══════════════════════════════════════════════════════════

def _build_definition():
    return {
        "DataSetIdentifierDeclarations": _dataset_declarations(),
        "Sheets": [_sheet2(), _sheet3(), _sheet4(), _sheet5()],
        "FilterGroups": [
            _filter_group("fg-campaign", "ad_campaign_id",     _DS_INFERENCE),
            _filter_group("fg-device",   "device_type",        _DS_INFERENCE),
            _filter_group("fg-category", "product_category",   _DS_INFERENCE),
            _filter_group("fg-segment",  "propensity_segment", _DS_INFERENCE),
        ],
    }


def _filter_group(fg_id, col_name, ds_alias):
    """Build a CategoryFilter group scoped to ALL_VISUALS on each relevant sheet."""
    return {
        "FilterGroupId": fg_id,
        "Filters": [{
            "CategoryFilter": {
                "FilterId": f"{fg_id}-filter",
                "Column": _col(ds_alias, col_name),
                "Configuration": {
                    "FilterListConfiguration": {
                        "MatchOperator": "CONTAINS",
                        "SelectAllOptions": "FILTER_ALL_VALUES",
                    }
                },
            }
        }],
        "ScopeConfiguration": {
            "SelectedSheets": {
                "SheetVisualScopingConfigurations": [
                    {"SheetId": sid, "Scope": "ALL_VISUALS"}
                    for sid in ["sheet-3", "sheet-4", "sheet-5"]
                ]
            }
        },
        "Status": "ENABLED",
        "CrossDataset": "SINGLE_DATASET",
    }


def _analysis_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeAnalysis",
            "quicksight:DescribeAnalysisPermissions",
            "quicksight:UpdateAnalysis",
            "quicksight:UpdateAnalysisPermissions",
            "quicksight:DeleteAnalysis",
            "quicksight:RestoreAnalysis",
            "quicksight:QueryAnalysis",
        ],
    }]


def _dashboard_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDashboard",
            "quicksight:ListDashboardVersions",
            "quicksight:UpdateDashboardPermissions",
            "quicksight:QueryDashboard",
            "quicksight:UpdateDashboard",
            "quicksight:DeleteDashboard",
            "quicksight:DescribeDashboardPermissions",
            "quicksight:UpdateDashboardPublishedVersion",
        ],
    }]


def ensure_dashboard(user_arn):
    """Create or update the QuickSight analysis and dashboard."""
    print("\n[6/6] Creating QuickSight analysis and dashboard...")

    definition = _build_definition()

    # ── Analysis ──
    try:
        qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        qs.update_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name="Customer Propensity Scoring",
            Definition=definition,
        )
        log(f"Updated analysis: {QS_ANALYSIS_ID}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name="Customer Propensity Scoring",
            Definition=definition,
            Permissions=_analysis_permissions(user_arn),
        )
        log(f"Created analysis: {QS_ANALYSIS_ID}")

    _wait_for_analysis()

    # ── Dashboard ──
    try:
        qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        qs.update_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name="Customer Propensity Scoring",
            Definition=definition,
            DashboardPublishOptions={
                "AdHocFilteringOption":  {"AvailabilityStatus": "ENABLED"},
                "ExportToCSVOption":     {"AvailabilityStatus": "ENABLED"},
                "VisualPublishOptions":  {"ExportHiddenFieldsOption": {"AvailabilityStatus": "DISABLED"}},
            },
        )
        log(f"Updated dashboard: {QS_DASHBOARD_ID}")
        # Publish the new version
        resp = qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        latest = resp["Dashboard"]["Version"]["VersionNumber"]
        qs.update_dashboard_published_version(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            VersionNumber=latest,
        )
        log(f"Published dashboard version: {latest}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name="Customer Propensity Scoring",
            Definition=definition,
            Permissions=_dashboard_permissions(user_arn),
            DashboardPublishOptions={
                "AdHocFilteringOption":  {"AvailabilityStatus": "ENABLED"},
                "ExportToCSVOption":     {"AvailabilityStatus": "ENABLED"},
                "VisualPublishOptions":  {"ExportHiddenFieldsOption": {"AvailabilityStatus": "DISABLED"}},
            },
        )
        log(f"Created dashboard: {QS_DASHBOARD_ID}")


def _wait_for_analysis(max_wait=120):
    for _ in range(max_wait // 5):
        resp = qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        status = resp["Analysis"]["Status"]
        if status in ("CREATION_SUCCESSFUL", "UPDATE_SUCCESSFUL"):
            log(f"Analysis status: {status}")
            return
        if "FAILED" in status:
            errors = resp["Analysis"].get("Errors", [])
            raise RuntimeError(f"Analysis failed ({status}): {errors}")
        time.sleep(5)
    print("  WARNING: Analysis did not reach SUCCESSFUL status within timeout.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AWS Clean Rooms ML — Create QuickSight Dashboard")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Output bucket: {OUTPUT_BUCKET}")

    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    ensure_quicksight_account()
    username = ensure_quicksight_user()
    user_arn = _qs_user_arn(username)

    prepare_glue_tables()
    ensure_quicksight_s3_access()
    ensure_datasource(user_arn)
    ensure_datasets(user_arn)
    ensure_dashboard(user_arn)

    dashboard_url = (
        f"https://{AWS_REGION}.quicksight.aws.amazon.com"
        f"/sn/dashboards/{QS_DASHBOARD_ID}"
    )
    print("\n" + "=" * 60)
    print("Dashboard ready!")
    print("=" * 60)
    print(f"\n  {dashboard_url}")
    print(f"\n  Note: SPICE datasets may still be ingesting.")
    print(f"  If visuals show 'No data', wait ~2 min and refresh.")
    print(f"\nNext: open the URL above in your browser.")


if __name__ == "__main__":
    main()
