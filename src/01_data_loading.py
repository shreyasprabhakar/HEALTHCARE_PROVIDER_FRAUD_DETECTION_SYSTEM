# 01_data_loading.py
# STEP 1: Data Loading & Basic Understanding

import pandas as pd

# -----------------------------
# File paths
# -----------------------------
BASE_PATH = "../data/"

LABELS_FILE = BASE_PATH + "Train-1542865627584.csv"
INPATIENT_FILE = BASE_PATH + "Train_Inpatientdata-1542865627584.csv"
OUTPATIENT_FILE = BASE_PATH + "Train_Outpatientdata-1542865627584.csv"
BENEFICIARY_FILE = BASE_PATH + "Train_Beneficiarydata-1542865627584.csv"

# -----------------------------
# Load datasets
# -----------------------------
labels_df = pd.read_csv(LABELS_FILE)
inpatient_df = pd.read_csv(INPATIENT_FILE)
outpatient_df = pd.read_csv(OUTPATIENT_FILE)
beneficiary_df = pd.read_csv(BENEFICIARY_FILE)

# -----------------------------
# Basic sanity checks
# -----------------------------
print("\n===== DATASET SHAPES =====")
print("Labels:", labels_df.shape)
print("Inpatient:", inpatient_df.shape)
print("Outpatient:", outpatient_df.shape)
print("Beneficiary:", beneficiary_df.shape)

print("\n===== LABEL DATA PREVIEW =====")
print(labels_df.head())

print("\n===== FRAUD LABEL DISTRIBUTION =====")
print(labels_df["PotentialFraud"].value_counts())

print("\n===== INPATIENT COLUMNS =====")
print(inpatient_df.columns.tolist())

print("\n===== OUTPATIENT COLUMNS =====")
print(outpatient_df.columns.tolist())

print("\n===== BENEFICIARY COLUMNS =====")
print(beneficiary_df.columns.tolist())
