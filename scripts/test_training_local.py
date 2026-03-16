# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Test training locally without Docker or AWS resources.
Simulates the SageMaker AI directory structure and runs train.py directly.
"""
import sys, os, shutil, subprocess, json

project_root = os.path.join(os.path.dirname(__file__), "..")
TEST_DIR = os.path.join(project_root, "local_test")


def main():
    print("=" * 60)
    print("Local Training Test (no AWS needed)")
    print("=" * 60)

    # Clean and recreate test directory structure
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    train_dir = os.path.join(TEST_DIR, "input", "data", "train")
    model_dir = os.path.join(TEST_DIR, "model")
    output_dir = os.path.join(TEST_DIR, "output", "data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Copy data files
    data_dir = os.path.join(project_root, "data")
    shutil.copy2(os.path.join(data_dir, "advertiser_engagement.csv"), train_dir)
    shutil.copy2(os.path.join(data_dir, "retailer_purchases.csv"), train_dir)

    # Run training
    print("\nRunning training locally...")
    train_script = os.path.join(project_root, "containers", "training", "train.py")
    result = subprocess.run([
        sys.executable, train_script,
        "--train_dir", train_dir,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--train_file_format", "csv",
    ])

    if result.returncode != 0:
        print("\nTraining FAILED")
        sys.exit(1)

    # Show results
    print("\n=== Model artifacts ===")
    for f in os.listdir(model_dir):
        size = os.path.getsize(os.path.join(model_dir, f))
        print(f"  {f}  ({size} bytes)")

    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print("\n=== Metrics ===")
        with open(metrics_path) as f:
            print(json.dumps(json.load(f), indent=2))

    print("\nDone!")


if __name__ == "__main__":
    main()
