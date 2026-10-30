
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AquaSense | Water Intelligence",
    page_icon="💧",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f8fc;
}
[data-testid="stSidebar"] {
    background-color: #e7f1fa;
}
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #123b60;
}
.subtitle {
    color: #60758a;
    font-size: 16px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #dce8f2;
}
.section-title {
    color: #123b60;
    font-size: 22px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
MODEL_PATH = Path(__file__).parent / "model" / "water_quality_model.pkl"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100:
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Model could not be loaded: {e}")
            return None
    return None

model_bundle = load_model()

# Supports either a saved model bundle or a standalone pipeline
if isinstance(model_bundle, dict):
    model = model_bundle.get("model")
    threshold = model_bundle.get("threshold", 0.40)
else:
    model = model_bundle
    threshold = 0.40

# ---------------- SIDEBAR ----------------
st.sidebar.title("💧 AquaSense")
st.sidebar.caption("Water Intelligence System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Water Prediction", "Analytics", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered water quality screening")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown('<p class="main-title">Water Quality Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Monitor water parameters and assess potability using machine learning.</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Water Parameters", "9")

    with col2:
        st.metric("Prediction Classes", "2")

    with col3:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    st.markdown("### Welcome to AquaSense")

    st.write("""
    AquaSense is a machine-learning-based water quality screening
    application designed to analyze important physicochemical
    properties of water samples.

    Enter water measurements to obtain a potability prediction,
    inspect probability outputs, and understand model uncertainty.
    """)

    st.info(
        "This tool provides preliminary screening only. "
        "It does not replace certified laboratory water testing."
    )

    st.markdown("### Water Quality Parameters")

    st.write(
        "pH • Hardness • Solids • Chloramines • Sulfate • "
        "Conductivity • Organic Carbon • Trihalomethanes • Turbidity"
    )

# ---------------- WATER PREDICTION ----------------
elif page == "Water Prediction":

    st.markdown('<p class="main-title">Water Potability Prediction</p>',
                unsafe_allow_html=True)

    st.write("Enter the physicochemical measurements of your water sample.")

    with st.form("water_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0)
            hardness = st.number_input("Hardness", 0.0, value=200.0)
            solids = st.number_input("Solids", 0.0, value=20000.0)

        with col2:
            chloramines = st.number_input("Chloramines", 0.0, value=7.0)
            sulfate = st.number_input("Sulfate", 0.0, value=300.0)
            conductivity = st.number_input("Conductivity", 0.0, value=400.0)

        with col3:
            organic_carbon = st.number_input("Organic Carbon", 0.0, value=14.0)
            trihalomethanes = st.number_input("Trihalomethanes", 0.0, value=60.0)
            turbidity = st.number_input("Turbidity", 0.0, value=4.0)

        submitted = st.form_submit_button(
            "Analyze Water Quality",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }])

        if model is None:
            st.error(
                "Model file not found. Place your trained model at "
                "model/water_quality_model.pkl to enable predictions."
            )

        else:
            try:
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                potable_index = classes.index(1)
                potable_probability = float(probabilities[potable_index])

                prediction = int(potable_probability >= threshold)

                st.markdown("---")
                st.markdown("### Prediction Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    if prediction == 1:
                        st.success("Classification: POTABLE")
                    else:
                        st.error("Classification: NOT POTABLE")

                with result_col2:
                    st.metric(
                        "Potable Probability",
                        f"{potable_probability * 100:.2f}%"
                    )

                # Binary predictive entropy
                p = np.clip(probabilities, 1e-10, 1.0)
                entropy = -np.sum(p * np.log(p)) / np.log(len(p))

                st.markdown("### Model Uncertainty")

                st.progress(float(entropy))
                st.write(f"Normalized probability entropy: {entropy:.3f}")

                if entropy >= 0.80:
                    st.warning("High uncertainty: additional testing is recommended.")
                elif entropy >= 0.50:
                    st.info("Moderate uncertainty: interpret the prediction carefully.")
                else:
                    st.success("Low probability uncertainty.")

                st.caption(
                    "Probability outputs are not necessarily calibrated "
                    "and do not certify that water is safe to drink."
                )

                st.markdown("### Class Probabilities")

                probability_df = pd.DataFrame({
                    "Class": [str(c) for c in classes],
                    "Probability (%)": probabilities * 100
                })

                st.bar_chart(
                    probability_df.set_index("Class")
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.markdown('<p class="main-title">Water Quality Analytics</p>',
                unsafe_allow_html=True)

    st.write("Explore the nine input parameters used by the prediction model.")

    sample_data = pd.DataFrame({
        "Parameter": [
            "pH", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic Carbon",
            "Trihalomethanes", "Turbidity"
        ],
        "Default Input": [
            7.0, 200, 20000, 7, 300, 400, 14, 60, 4
        ]
    })

    st.bar_chart(sample_data.set_index("Parameter"))
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.caption(
        "These are illustrative input values, not measured population statistics."
    )

# ---------------- ABOUT ----------------
elif page == "About Project":

    st.markdown('<p class="main-title">About AquaSense</p>',
                unsafe_allow_html=True)

    st.write("""
    **Project:** Bayesian Water Quality Intelligence and
    Uncertainty-Based Potability Prediction.

    **Objective:** To build a machine-learning-based screening
    system that evaluates water quality using physicochemical
    parameters.

    **Core concepts:**
    - Bayesian machine learning
    - Probability-based prediction
    - Uncertainty estimation
    - Data preprocessing
    - Interactive visualization

    **Technology stack:**
    Python, Scikit-learn, Pandas, NumPy and Streamlit.
    """)

    st.info(
        "AquaSense is an educational decision-support prototype, "
        "not a substitute for regulatory water-quality certification."
    )
