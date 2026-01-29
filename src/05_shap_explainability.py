# 05_shap_explainability.py
# STEP 5: SHAP Explainability

import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
DATA_FILE = "../outputs/provider_features.csv"
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"].map({"Yes": 1, "No": 0})

# -----------------------------
# Load model and scaler
# -----------------------------
with open("../outputs/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../outputs/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

# Use a sample for speed
X_sample = pd.DataFrame(
    X_scaled,
    columns=X.columns
).sample(200, random_state=42)

shap_values = explainer.shap_values(X_sample)

# -----------------------------
# Global explanation
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("../outputs/shap_summary.png", dpi=300)
plt.close()

print("SHAP summary plot saved.")

# -----------------------------
# Individual explanation (one fraud case)
# -----------------------------
fraud_index = y[y == 1].index[0]
X_fraud = pd.DataFrame(
    X_scaled[fraud_index].reshape(1, -1),
    columns=X.columns
)

shap_values_single = explainer.shap_values(X_fraud)

plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=X_fraud.iloc[0],
        feature_names=X.columns
    ),
    show=False
)
plt.tight_layout()
plt.savefig("../outputs/shap_individual.png", dpi=300)
plt.close()

print("SHAP individual explanation saved.")
