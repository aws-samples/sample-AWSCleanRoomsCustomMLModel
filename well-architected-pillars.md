# Well-Architected Pillars — Guidance for Predicting Purchase Intent with AWS Clean Rooms ML

## Operational Excellence

This Guidance automates the end-to-end ML pipeline through six sequential Python scripts, each idempotent and safe to re-run without duplicating resources. A unique run ID isolates each deployment, enabling parallel experimentation and clean teardown. [AWS CodeBuild](https://aws.amazon.com/codebuild/) manages container image builds without requiring local Docker installations, while [Amazon QuickSight](https://aws.amazon.com/quicksight/) dashboards provide immediate operational visibility into model outputs. Two undeploy scripts (region scan + resource cleanup) ensure complete environment teardown in reverse dependency order, reducing operational overhead.

## Security

The Guidance applies least-privilege IAM principles throughout: all roles use scoped inline policies restricted to specific bucket ARNs and prefixes, with no `*FullAccess` managed policies attached. [Amazon S3](https://aws.amazon.com/s3/) buckets enforce Block Public Access, SSE-S3 encryption by default, versioning for audit trails, and TLS-only access via `aws:SecureTransport` condition keys. [AWS Clean Rooms](https://aws.amazon.com/clean-rooms/) enforces that neither party accesses the other's raw data; the join happens within the secure collaboration boundary without raw records crossing it. Container input validation enforces a 50 MB payload size limit and schema checks, and no credentials are hardcoded in any script.

## Reliability

Each deployment script is idempotent: if a resource already exists, it is reused rather than duplicated, enabling safe recovery from interrupted deployments. AWS Clean Rooms ML orchestration includes explicit state polling (waiting for channels, models, and inference jobs to reach ACTIVE status) with graceful handling of intermediate states. S3 versioning preserves data history for recovery, and ML input channels are configured with 30-day retention periods ensuring data availability for retraining or auditing. The separation of source and output buckets prevents accidental overwrites of input data by inference results.

## Performance Efficiency

The architecture leverages managed services to eliminate infrastructure management: [AWS Clean Rooms ML](https://aws.amazon.com/clean-rooms/ml/) handles compute scaling for training and inference jobs, [Amazon ECR](https://aws.amazon.com/ecr/) serves container images with low-latency pulls, and [Amazon Athena](https://aws.amazon.com/athena/) provides serverless query execution over inference results. The GradientBoosting model (100 estimators, max_depth 5) is deliberately sized for fast training on the joined dataset, and the inference container produces scores for ~40,000 records in a single batch run. Container images use optimized base images (SageMaker AI PyTorch) to minimize cold-start time.

## Cost Optimization

This Guidance minimizes cost through a fully serverless, pay-per-use architecture. [AWS Clean Rooms](https://aws.amazon.com/clean-rooms/) charges only for queries executed; there are no idle compute resources between runs. S3 storage costs scale linearly with data volume (synthetic demo data is under 5 MB). CodeBuild charges only for build minutes (~7 minutes per container build). Amazon QuickSight SPICE datasets cache results to avoid repeated Athena queries. The undeploy scripts ensure complete resource cleanup, preventing ongoing charges from forgotten infrastructure. All bucket names include the account ID and run ID, making cost attribution straightforward.

## Sustainability

The Guidance reduces environmental impact by using fully managed, serverless services that share underlying infrastructure across customers, eliminating dedicated idle resources. Containers are built once and reused across training and inference cycles; the inference image uses the same base as training, minimizing redundant storage. The synthetic data generation approach (seed-deterministic, local execution) avoids unnecessary cloud compute for test data creation. The architecture's clean teardown capability ensures no orphaned resources consume energy after experimentation concludes.
