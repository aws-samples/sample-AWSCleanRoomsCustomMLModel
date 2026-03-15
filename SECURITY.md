<!-- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!-- SPDX-License-Identifier: MIT-0 -->

## Security

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

## Shared Responsibility Model

This demo deploys resources in your AWS account. Security and compliance are a [shared responsibility](https://aws.amazon.com/compliance/shared-responsibility-model/) between AWS and you. AWS is responsible for the security *of* the cloud (infrastructure, managed services). You are responsible for security *in* the cloud — including IAM policies, data encryption settings, network configuration, and access controls applied to the resources this demo creates.

## Security Implementation in This Demo

The following security controls are implemented in the setup scripts:

### IAM — Least-Privilege Policies

- All IAM roles use scoped inline policies rather than AWS managed `*FullAccess` policies.
- S3 permissions are restricted to the specific demo buckets and prefixes.
- AWS Glue permissions are scoped to the demo database and tables.
- ECR permissions separate `GetAuthorizationToken` (requires `Resource: *`) from push/pull actions (scoped to specific repository ARNs).
- Each role has a dedicated trust policy limited to the specific AWS service that assumes it.

### S3 Bucket Hardening

- **Block Public Access** is enabled on all buckets (all four BPA settings set to `true`).
- **Server-Side Encryption** (SSE-S3, AES-256) is enabled by default.
- **Versioning** is enabled for data integrity and recovery.
- **TLS-only bucket policy** denies any request where `aws:SecureTransport` is `false`.

### Container Security

- Inference container enforces a **maximum input payload size** (50 MB) to prevent resource exhaustion.
- Inference container validates the **input schema** (expected column names) before processing.
- Training container wraps CSV/Parquet parsing in error handling to surface clear failure messages.
- Both containers use the official Amazon SageMaker AI base images from Amazon ECR.

### Configuration

- `config.py` reads `AWS_ACCOUNT_ID` and `AWS_REGION` from environment variables, avoiding hardcoded credentials.
- Subprocess calls in `build_and_push.py` use `shell=False` to prevent shell injection.

## Threat Model — Key Risks

| Threat | Mitigation |
|--------|------------|
| Over-privileged IAM roles | Inline policies scoped to specific ARNs; no `*FullAccess` managed policies |
| Public S3 bucket exposure | Block Public Access enabled; TLS-only bucket policy |
| Data exfiltration via containers | AWS Clean Rooms ML runs containers in an isolated environment; output is written to a configured S3 bucket only |
| Malicious input to inference endpoint | Payload size limit, schema validation, error handling |
| Credential leakage in code | Account ID and region read from environment variables; no secrets in source |
| Shell injection via subprocess | All subprocess calls use `shell=False` with argument lists |

## Security Review Checklist

Before deploying to a production or shared environment:

1. **Rotate credentials** — Ensure the AWS credentials used during setup are rotated or scoped to a temporary session.
2. **Review IAM policies** — Verify the inline policies created by `setup_cleanrooms.py` match your organization's requirements.
3. **Enable CloudTrail** — Ensure AWS CloudTrail is logging API calls in the region where you deploy.
4. **Enable GuardDuty** — Enable Amazon GuardDuty for threat detection on the account.
5. **Restrict network access** — If running containers locally, ensure Docker is not exposing ports to the public internet.
6. **Scan container images** — Run Amazon ECR image scanning on the pushed training and inference images.
7. **Review S3 bucket policies** — Confirm the TLS-only and Block Public Access settings are active after deployment.
