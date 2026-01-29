# 03_model_training.py
# STEP 3: Model Training (XGBoost)

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

# -----------------------------
# Load processed data
# -----------------------------
DATA_FILE = "../outputs/provider_features.csv"
df = pd.read_csv(DATA_FILE)

# -----------------------------
# Prepare features and target
# -----------------------------
X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"].map({"Yes": 1, "No": 0})

# -----------------------------
# Train-validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -----------------------------
# Handle class imbalance
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# XGBoost model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Train model
# -----------------------------
model.fit(X_train_scaled, y_train)

# -----------------------------
# Quick sanity check (AUROC)
# -----------------------------
y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)

print(f"Validation AUROC: {auc:.3f}")

# -----------------------------
# Save model and scaler
# -----------------------------
with open("../outputs/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../outputs/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
