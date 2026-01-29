# ui/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")

# -------------------------------------------------
# SESSION STATE INITIALIZATION  (STEP 1)
# -------------------------------------------------
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "provider_df" not in st.session_state:
    st.session_state.provider_df = None

if "X" not in st.session_state:
    st.session_state.X = None

if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None

if "model" not in st.session_state:
    st.session_state.model = None


# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("Healthcare Provider Fraud Detection System")
st.write(
    """
    This application analyzes healthcare claim data and ranks providers
    based on fraud risk. It is designed as a **decision-support tool**
    for investigators, not an automated verdict system.
    """
)

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Input Options")

use_sample_data = st.sidebar.checkbox(
    "Use sample data from project /data folder"
)

labels_file = st.sidebar.file_uploader("Provider Labels CSV", type=["csv"])
inpatient_file = st.sidebar.file_uploader("Inpatient Claims CSV", type=["csv"])
outpatient_file = st.sidebar.file_uploader("Outpatient Claims CSV", type=["csv"])

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Fraud Risk Threshold", 0.0, 1.0, 0.5, 0.05
)

# -------------------------------------------------
# Feature engineering (MATCHES TRAINING)
# -------------------------------------------------
def build_provider_features(labels_df, inpatient_df, outpatient_df):

    inpatient_agg = inpatient_df.groupby("Provider").agg(
        inpatient_claims=("ClaimID", "count"),
        inpatient_total_reimbursed=("InscClaimAmtReimbursed", "sum"),
        inpatient_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
        inpatient_std_reimbursed=("InscClaimAmtReimbursed", "std"),
        inpatient_total_deductible=("DeductibleAmtPaid", "sum"),
    ).reset_index()

    outpatient_agg = outpatient_df.groupby("Provider").agg(
        outpatient_claims=("ClaimID", "count"),
        outpatient_total_reimbursed=("InscClaimAmtReimbursed", "sum"),
        outpatient_avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
        outpatient_std_reimbursed=("InscClaimAmtReimbursed", "std"),
    ).reset_index()

    provider_df = labels_df.merge(inpatient_agg, on="Provider", how="left")
    provider_df = provider_df.merge(outpatient_agg, on="Provider", how="left")
    provider_df.fillna(0, inplace=True)

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

    return provider_df


# -------------------------------------------------
# RUN FRAUD DETECTION  (STEP 2 + STEP 3)
# -------------------------------------------------
if st.button("Run Fraud Detection"):

    with st.spinner("Running fraud detection pipeline..."):

        # Load data
        if use_sample_data:
            labels_df = pd.read_csv("../data/Train-1542865627584.csv")
            inpatient_df = pd.read_csv("../data/Train_Inpatientdata-1542865627584.csv")
            outpatient_df = pd.read_csv("../data/Train_Outpatientdata-1542865627584.csv")
            st.info("Using sample data from project /data folder.")
        else:
            if not (labels_file and inpatient_file and outpatient_file):
                st.error("Please upload all datasets or select 'Use sample data'.")
                st.stop()

            labels_df = pd.read_csv(labels_file)
            inpatient_df = pd.read_csv(inpatient_file)
            outpatient_df = pd.read_csv(outpatient_file)

        # Feature engineering
        provider_df = build_provider_features(
            labels_df, inpatient_df, outpatient_df
        )

        # Load model & scaler
        with open("../outputs/xgb_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("../outputs/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Build X
        X = provider_df.drop(columns=["Provider", "PotentialFraud"])
        X_scaled = scaler.transform(X)

        # Predict
        provider_df["fraud_score"] = model.predict_proba(X_scaled)[:, 1]
        provider_df["flagged"] = provider_df["fraud_score"] >= threshold
        provider_df.sort_values("fraud_score", ascending=False, inplace=True)

        # -------- STORE IN SESSION STATE (STEP 3) --------
        st.session_state.provider_df = provider_df
        st.session_state.X = X
        st.session_state.X_scaled = X_scaled
        st.session_state.model = model
        st.session_state.results_ready = True

    st.success("Fraud detection completed.")

# -------------------------------------------------
# DISPLAY RESULTS  (STEP 4)
# -------------------------------------------------
if st.session_state.results_ready:

    provider_df = st.session_state.provider_df
    X = st.session_state.X
    X_scaled = st.session_state.X_scaled
    model = st.session_state.model

    st.subheader("Ranked Providers by Fraud Risk")

    display_df = provider_df[["Provider", "fraud_score", "flagged"]].copy()
    display_df["fraud_score"] = display_df["fraud_score"].round(3)
    display_df["Risk Level"] = np.where(
        display_df["flagged"], "High Risk", "Low / Medium Risk"
    )

    st.dataframe(display_df, use_container_width=True)

    # Export button
    st.download_button(
        label="Download Results as CSV",
        data=display_df.to_csv(index=False),
        file_name="fraud_risk_results.csv",
        mime="text/csv"
    )

    # -------------------------------------------------
    # SHAP EXPLAINABILITY
    # -------------------------------------------------
    st.subheader("Explain a Provider (SHAP)")

    selected_provider = st.selectbox(
        "Select Provider ID",
        provider_df["Provider"].unique()
    )

    idx = provider_df[
        provider_df["Provider"] == selected_provider
    ].index[0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    st.markdown(
        f"**Fraud Score:** {provider_df.loc[idx, 'fraud_score']:.3f}"
    )

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X.iloc[idx],
            feature_names=X.columns
        ),
        show=False
    )
    st.pyplot(fig)

    st.info(
        "SHAP explains how each feature contributed to the provider’s fraud risk score. "
        "Final investigation decisions should always be made by human experts."
    )

