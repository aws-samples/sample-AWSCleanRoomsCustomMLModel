# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Test training locally without Docker
# No AWS config needed — runs purely on local data

$TEST_DIR = "local_test"

if (Test-Path $TEST_DIR) { Remove-Item -Recurse -Force $TEST_DIR }

New-Item -ItemType Directory -Path "$TEST_DIR/input/data/train" -Force | Out-Null
New-Item -ItemType Directory -Path "$TEST_DIR/model" -Force | Out-Null
New-Item -ItemType Directory -Path "$TEST_DIR/output/data" -Force | Out-Null

Copy-Item "data/advertiser_engagement.csv" "$TEST_DIR/input/data/train/"
Copy-Item "data/retailer_purchases.csv" "$TEST_DIR/input/data/train/"

Write-Output "Running training locally..."
python containers/training/train.py `
    --train_dir "$TEST_DIR/input/data/train" `
    --model_dir "$TEST_DIR/model" `
    --output_dir "$TEST_DIR/output/data" `
    --train_file_format csv

Write-Output ""
Write-Output "=== Model artifacts ==="
Get-ChildItem "$TEST_DIR/model" | Format-Table Name, Length

Write-Output "=== Metrics ==="
Get-Content "$TEST_DIR/output/data/metrics.json"
