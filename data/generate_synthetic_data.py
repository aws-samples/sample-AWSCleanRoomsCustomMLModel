# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate synthetic demo data for AWS Clean Rooms Custom ML Model demo.

Scenario: Customer Propensity Scoring
- Party A (Advertiser): ad engagement data
- Party B (Retailer): purchase transaction data
- Shared key: user_id (overlapping user space)
"""

import csv
import random  # nosec B311 — synthetic data generation; cryptographic randomness not required
import os
from datetime import datetime, timedelta

random.seed(42)

NUM_USERS = 10000
# 80% overlap — some users exist in only one dataset
SHARED_USERS = int(NUM_USERS * 0.8)
ADVERTISER_ONLY = int(NUM_USERS * 0.1)
RETAILER_ONLY = int(NUM_USERS * 0.1)

shared_user_ids = [f"user_{i:06d}" for i in range(SHARED_USERS)]
advertiser_only_ids = [f"user_{SHARED_USERS + i:06d}" for i in range(ADVERTISER_ONLY)]
retailer_only_ids = [f"user_{SHARED_USERS + ADVERTISER_ONLY + i:06d}" for i in range(RETAILER_ONLY)]

advertiser_user_ids = shared_user_ids + advertiser_only_ids
retailer_user_ids = shared_user_ids + retailer_only_ids

CAMPAIGNS = ["camp_summer_sale", "camp_back_to_school", "camp_holiday", "camp_spring", "camp_clearance"]
DEVICES = ["mobile", "desktop", "tablet", "smart_tv"]
CATEGORIES = ["electronics", "clothing", "home_garden", "sports", "beauty", "grocery", "toys"]

BASE_DATE = datetime(2025, 1, 1)


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


USER_PROPENSITY = {}


def generate_advertiser_data():
    rows = []
    for uid in advertiser_user_ids:
        num_campaigns = random.randint(1, 3)
        propensity = USER_PROPENSITY.get(uid, random.gauss(0.4, 0.2))
        propensity = max(0.05, min(0.95, propensity))

        for campaign in random.sample(CAMPAIGNS, num_campaigns):
            impressions = max(1, int(random.randint(1, 30) + 30 * propensity))
            device = random.choice(DEVICES)
            base_ctr = {"mobile": 0.08, "desktop": 0.05, "tablet": 0.06, "smart_tv": 0.03}[device]
            ctr = base_ctr * (0.2 + 2.0 * propensity) * random.uniform(0.8, 1.3)
            clicks = max(0, int(impressions * ctr))
            time_per_click = random.uniform(5, 30) * (0.2 + 2.0 * propensity)
            time_spent = round(clicks * time_per_click, 1) if clicks > 0 else 0
            event_date = random_date(BASE_DATE, BASE_DATE + timedelta(days=180))

            rows.append({
                "user_id": uid,
                "ad_campaign_id": campaign,
                "impressions": impressions,
                "clicks": clicks,
                "time_spent_seconds": time_spent,
                "device_type": device,
                "event_date": event_date.strftime("%Y-%m-%d"),
            })
    return rows


def generate_retailer_data():
    rows = []
    shared_set = set(shared_user_ids)
    for uid in retailer_user_ids:
        is_shared = uid in shared_set

        if is_shared:
            base_propensity = random.gauss(0.58, 0.15)
        else:
            base_propensity = random.gauss(0.32, 0.15)
        base_propensity = max(0.05, min(0.95, base_propensity))

        USER_PROPENSITY[uid] = base_propensity

        noise = random.gauss(0, 0.08)
        converted = (base_propensity + noise) > 0.50

        num_categories = random.randint(1, 4)
        for category in random.sample(CATEGORIES, num_categories):
            site_visits = max(1, int(random.randint(1, 8) + 15 * base_propensity))

            if converted:
                purchase_count = random.randint(1, 15)
                avg_price = {"electronics": 150, "clothing": 45, "home_garden": 65,
                             "sports": 55, "beauty": 30, "grocery": 25, "toys": 35}[category]
                purchase_amount = round(purchase_count * avg_price * random.uniform(0.5, 1.5), 2)
                days_since = random.randint(1, 180)
            else:
                purchase_count = 0
                purchase_amount = 0.0
                days_since = random.randint(1, 180)

            last_date = (BASE_DATE + timedelta(days=180) - timedelta(days=min(days_since, 180))).strftime("%Y-%m-%d")

            rows.append({
                "user_id": uid,
                "product_category": category,
                "purchase_amount": purchase_amount,
                "purchase_count": purchase_count,
                "site_visits": site_visits,
                "days_since_last_purchase": days_since,
                "last_purchase_date": last_date,
                "converted": int(converted),
            })
    return rows


def write_csv(filename, rows, fieldnames):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows to {filename}")


if __name__ == "__main__":
    print("Generating retailer data (Party B)...")
    ret_data = generate_retailer_data()
    write_csv(
        "data/retailer_purchases.csv",
        ret_data,
        ["user_id", "product_category", "purchase_amount", "purchase_count", "site_visits", "days_since_last_purchase", "last_purchase_date", "converted"],
    )

    print("Generating advertiser data (Party A)...")
    adv_data = generate_advertiser_data()
    write_csv(
        "data/advertiser_engagement.csv",
        adv_data,
        ["user_id", "ad_campaign_id", "impressions", "clicks", "time_spent_seconds", "device_type", "event_date"],
    )

    adv_users = set(r["user_id"] for r in adv_data)
    ret_users = set(r["user_id"] for r in ret_data)
    overlap = adv_users & ret_users
    print(f"\nStats:")
    print(f"  Advertiser users: {len(adv_users)}")
    print(f"  Retailer users:   {len(ret_users)}")
    print(f"  Overlapping:      {len(overlap)}")
    print(f"  Advertiser rows:  {len(adv_data)}")
    print(f"  Retailer rows:    {len(ret_data)}")
