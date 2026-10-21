"""
Streamlit dashboard for the Churn Early-Warning System.
Run: streamlit run dashboard/app.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from predict import predict_with_risk, load_model
from explain import get_feature_importance

st.set_page_config(
    page_title="Churn Early-Warning",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Churn Early-Warning System")
st.caption("Identify high-risk customers and understand the drivers of churn")

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("High-risk threshold", 0.3, 0.9, 0.5, 0.05)

# Check model
try:
    model = load_model()
    model_ok = True
except FileNotFoundError:
    model_ok = False
    st.error("Model not found. Please run `python src/train_model.py` first.")
    st.stop()

# Load processed data if available
processed_path = ROOT / "data" / "processed" / "telco_clean.csv"
if processed_path.exists():
    df = pd.read_csv(processed_path)
    features = df.drop(columns=["Churn"], errors="ignore")
    scored = predict_with_risk(features, threshold=threshold)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(scored):,}")
    col2.metric("High Risk", f"{(scored['risk_label'] == 'High').sum():,}")
    col3.metric("Avg Churn Prob", f"{scored['churn_probability'].mean():.1%}")
    if "Churn" in df.columns:
        col4.metric("Actual Churn Rate", f"{df['Churn'].mean():.1%}")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["High-Risk Customers", "Feature Importance", "Score a Customer"])

    with tab1:
        high_risk = scored[scored["risk_label"] == "High"].head(50)
        st.subheader(f"Top {len(high_risk)} High-Risk Customers")
        st.dataframe(
            high_risk[["churn_probability", "risk_label"] + [c for c in high_risk.columns if c not in ("churn_probability", "risk_label")]].head(30),
            use_container_width=True,
        )

        fig = px.histogram(
            scored,
            x="churn_probability",
            nbins=30,
            title="Distribution of Churn Probability",
            color_discrete_sequence=["#ef4444"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top Risk Drivers")
        try:
            imp = get_feature_importance(top_n=15)
            fig2 = px.bar(
                imp,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance",
                color="importance",
                color_continuous_scale="Reds",
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")

    with tab3:
        st.subheader("Score a Single Customer")
        with st.form("score_form"):
            c1, c2 = st.columns(2)
            tenure = c1.number_input("Tenure (months)", min_value=0, value=12)
            monthly = c1.number_input("Monthly Charges", min_value=0.0, value=70.0)
            total = c1.number_input("Total Charges", min_value=0.0, value=850.0)
            contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = c2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment = c2.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([{
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "Contract": contract,
                "InternetService": internet,
                "PaymentMethod": payment,
            }])
            # Fill other columns with defaults from training data if needed
            for col in features.columns:
                if col not in row.columns:
                    row[col] = features[col].mode().iloc[0] if features[col].dtype == object else features[col].median()

            row = row[features.columns]  # same order
            result = predict_with_risk(row, threshold=threshold)
            prob = result["churn_probability"].iloc[0]
            label = result["risk_label"].iloc[0]

            st.metric("Churn Probability", f"{prob:.1%}")
            if label == "High":
                st.error(f"Risk Level: **{label}** — Consider retention actions.")
            else:
                st.success(f"Risk Level: **{label}**")
else:
    st.info(
        "No processed data found. Place the Telco dataset in `data/raw/telco_churn.csv` "
        "and run `python src/train_model.py`."
    )
