# 02_feature_engineering.py
# STEP 2: Provider-Level Feature Engineering

import pandas as pd
import numpy as np

# -----------------------------
# File paths
# -----------------------------
BASE_PATH = "../data/"

LABELS_FILE = BASE_PATH + "Train-1542865627584.csv"
INPATIENT_FILE = BASE_PATH + "Train_Inpatientdata-1542865627584.csv"
OUTPATIENT_FILE = BASE_PATH + "Train_Outpatientdata-1542865627584.csv"

# -----------------------------
# Load datasets
# -----------------------------
labels_df = pd.read_csv(LABELS_FILE)
inpatient_df = pd.read_csv(INPATIENT_FILE)
outpatient_df = pd.read_csv(OUTPATIENT_FILE)

# -----------------------------
# Inpatient aggregation
# -----------------------------
inpatient_agg = inpatient_df.groupby("Provider").agg(
    inpatient_claims=("ClaimID", "count"),
    inpatient_total_reimbursed=("InscClaimAmtReimbursed", "sum"),
    inpatient_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
    inpatient_std_reimbursed=("InscClaimAmtReimbursed", "std"),
    inpatient_total_deductible=("DeductibleAmtPaid", "sum"),
).reset_index()

# -----------------------------
# Outpatient aggregation
# -----------------------------
outpatient_agg = outpatient_df.groupby("Provider").agg(
    outpatient_claims=("ClaimID", "count"),
    outpatient_total_reimbursed=("InscClaimAmtReimbursed", "sum"),
    outpatient_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
    outpatient_std_reimbursed=("InscClaimAmtReimbursed", "std"),
).reset_index()

# -----------------------------
# Merge all provider-level data
# -----------------------------
provider_df = labels_df.merge(inpatient_agg, on="Provider", how="left")
provider_df = provider_df.merge(outpatient_agg, on="Provider", how="left")

# Replace NaN with 0 (providers with no inpatient or outpatient claims)
provider_df.fillna(0, inplace=True)

# -----------------------------
# Derived features
# -----------------------------
provider_df["total_claims"] = (
    provider_df["inpatient_claims"] + provider_df["outpatient_claims"]
)

provider_df["total_reimbursed"] = (
    provider_df["inpatient_total_reimbursed"]
    + provider_df["outpatient_total_reimbursed"]
)

provider_df["avg_reimbursed_per_claim"] = (
    provider_df["total_reimbursed"] / (provider_df["total_claims"] + 1)
)

provider_df["inpatient_ratio"] = (
    provider_df["inpatient_claims"] / (provider_df["total_claims"] + 1)
)

provider_df["cv_inpatient"] = (
    provider_df["inpatient_std_reimbursed"]
    / (provider_df["inpatient_avg_reimbursed"] + 1)
)

provider_df["cv_outpatient"] = (
    provider_df["outpatient_std_reimbursed"]
    / (provider_df["outpatient_avg_reimbursed"] + 1)
)

# -----------------------------
# Save processed dataset
# -----------------------------
OUTPUT_FILE = "../outputs/provider_features.csv"
provider_df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering completed.")
print("Final dataset shape:", provider_df.shape)
print("Fraud label distribution:")
print(provider_df["PotentialFraud"].value_counts())
