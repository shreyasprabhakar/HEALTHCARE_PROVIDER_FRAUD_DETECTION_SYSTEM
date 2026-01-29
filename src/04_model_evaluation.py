# 04_model_evaluation.py
# STEP 4: Model Evaluation & Testing

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score
)

# -----------------------------
# Load data
# -----------------------------
DATA_FILE = "../outputs/provider_features.csv"
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"].map({"Yes": 1, "No": 0})

# -----------------------------
# Train-validation split (same as training)
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load trained model and scaler
# -----------------------------
with open("../outputs/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../outputs/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_val_scaled = scaler.transform(X_val)

# -----------------------------
# Predictions
# -----------------------------
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

# Default threshold = 0.5
threshold = 0.5
y_val_pred = (y_val_proba >= threshold).astype(int)

# -----------------------------
# Metrics
# -----------------------------
auc = roc_auc_score(y_val, y_val_proba)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)

print(f"AUROC: {auc:.3f}")
print(f"Precision (threshold=0.5): {precision:.3f}")
print(f"Recall (threshold=0.5): {recall:.3f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_val, y_val_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUROC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Provider Fraud Detection")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("../outputs/roc_curve.png", dpi=300)
plt.close()

print("\nROC curve saved to outputs/roc_curve.png")
