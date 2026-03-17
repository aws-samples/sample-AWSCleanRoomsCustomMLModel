# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Scan multiple AWS regions for leftover cleanrooms-ml-demo collaborations.
Useful for finding resources deployed in regions other than the one in config.py.

Run with: python scripts/undeploy/scan_regions.py  (from the project root folder)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import *
validate()

import boto3

# Regions where AWS Clean Rooms is available
REGIONS = [
    "us-east-1", "us-east-2", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-north-1", "eu-central-1",
    "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ap-northeast-2",
]

def main():
    print("=" * 60)
    print("Scanning regions for Clean Rooms ML demo resources...")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}\n")

    found = []
    for r in REGIONS:
        try:
            c = boto3.client("cleanrooms", region_name=r)
            collabs = c.list_collaborations(memberStatus="ACTIVE").get("collaborationList", [])
            matches = [x for x in collabs if PREFIX in x.get("name", "")]
            if matches:
                print(f"  {r}: FOUND {len(matches)} collaboration(s)")
                for m in matches:
                    print(f"    - {m['name']} (id={m['id']})")
                found.append(r)
            else:
                print(f"  {r}: clean")
        except Exception as e:
            err = str(e)
            if "not available" in err.lower() or "endpoint" in err.lower():
                print(f"  {r}: (service not available)")
            else:
                print(f"  {r}: error - {e}")

    print()
    if found:
        print(f"Resources found in: {', '.join(found)}")
        print("Run undeploy for each region:")
        for r in found:
            print(f"  AWS_REGION={r} python scripts/undeploy/undeploy.py")
    else:
        print("All regions clean — no demo resources found.")


if __name__ == "__main__":
    main()
