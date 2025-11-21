import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os
# ---------- CONFIG ----------
# Put your model / preprocessors in the same folder as this script
MODEL_PATH = "disease_risk_model.pkl"     # your trained classifier (joblib)
ENCODER_PATH = "le.pkl"         # optional (for categorical label encoders)
SCALER_PATH = "scaler.pkl"           # optional (for numeric scaling)
FEATURE_ORDER = [
   'age', 'gender', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l',
       'calories_consumed', 'smoker', 'alcohol', 'resting_hr', 'systolic_bp',
       'diastolic_bp', 'cholesterol', 'family_history'
]
# Adjust FEATURE_ORDER to match the exact order your model expects.


@st.cache_resource
def load_artifacts():
    artifacts = {}
    # load model
    if os.path.exists("disease_risk_model.pkl"):
        artifacts["model"] = joblib.load("disease_risk_model.pkl")
    else:
        artifacts["model"] = None

    # optional encoder & scaler
    artifacts["encoder"] = joblib.load("le.pkl") if os.path.exists("le.pkl") else None
    artifacts["scaler"] = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return artifacts

def prepare_input(values: dict, artifacts):
   
    row = []
    for f in FEATURE_ORDER:
        v = values.get(f)
        # handle categorical encoding if encoder provided and field is categorical
        if artifacts.get("encoder") and f in getattr(artifacts["encoder"], "feature_names_in_", []):
            # assume encoder has transform that accepts list of values
            try:
                v = artifacts["encoder"].transform([[v]])[0][0]
            except Exception:
                pass
        row.append(v)

    X = np.array(row, dtype=float).reshape(1, -1)

    if artifacts.get("scaler"):
        try:
            X = artifacts["scaler"].transform(X)
        except Exception:
            # scaler failed — fall through with raw X
            pass

    return X

# ---------- APP ----------
st.set_page_config(page_title="Disease Risk Predictor", layout="centered")
st.title("Disease Risk Predictor — Streamlit App")
st.markdown("A simple UI to predict disease risk using your trained model. Edit `FEATURE_ORDER` in the script to match your model's expected input order.")

# optional image (uncomment and update path if you want)
# st.image(Image.open("your_banner.jpg"), use_column_width=True)

# Sidebar info
st.sidebar.header("Model / Deployment Info")
st.sidebar.write("Place these files in the same folder:")
st.sidebar.write(f"- `{"disease_risk_model.pkl"} (required)")
st.sidebar.write(f"- `{"le.pkl"}` (optional)")
st.sidebar.write(f"- `{"SCALER"}` (optional)")
st.sidebar.write("Update `FEATURE_ORDER` to match training features exactly.")

artifacts = load_artifacts()

if artifacts["model"] is None:
    st.error(f"Model file not found: {"disease_risk_model.pkl"}. Save your trained model with joblib and place it here.")
    st.stop()

model = artifacts["model"]

# --- User Inputs ---
st.header("Patient Input")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["male", "female", "other"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, format="%.1f")
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)
    sleep_hours = st.number_input("Sleep Hours per Day", min_value=0, max_value=24, value=7)
    water_intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=10.0, value=2.0)

with col2:
    calories_consumed = st.number_input("Calories Consumed", min_value=500, max_value=6000, value=2000)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["no", "yes"])
    resting_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=200, value=70)
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=240, value=120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=80)
    cholesterol = st.selectbox("Cholesterol level", ["normal", "above_normal", "high"])
    family_history = st.selectbox("Family History of Disease", ["no", "yes"])

# Quick display so user sees order
 
gender_map = {"male": 0, "female": 1, "other": 2}
smoker_map = {"no": 0, "yes": 1}
alcohol_map = {"no": 0, "yes": 1}
chol_map = {"normal": 0, "above_normal": 1, "high": 2}
family_history_map = {"no": 0, "yes": 1}

# --- Final numeric inputs ---
input_values = {
    "age": float(age),
    "gender": gender_map[gender],
    "bmi": float(bmi),
    "daily_steps": float(daily_steps),
    "sleep_hours": float(sleep_hours),
    "water_intake_l": float(water_intake),
    "calories_consumed": float(calories_consumed),
    "smoker": smoker_map[smoker],
    "alcohol": alcohol_map[alcohol],
    "resting_hr": float(resting_hr),
    "systolic_bp": float(systolic_bp),
    "diastolic_bp": float(diastolic_bp),
    "cholesterol": chol_map[cholesterol],
    "family_history": family_history_map[family_history]
}
# Predict button
if st.button("Predict Risk"):
    try:
        # Prepare input
        X = prepare_input(input_values, artifacts)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]

            # ✅ Get probability for class "1" (disease)
            try:
                positive_idx = list(model.classes_).index(1)
            except ValueError:
                positive_idx = 0  # fallback if model doesn't have class 1

            risk_score = proba[positive_idx]

            # Show probability
            st.metric("Estimated Risk Probability", f"{risk_score:.2%}")

            # Risk categories
            if risk_score >= 0.30:
                st.error("High risk — please consult a clinician.")
            elif risk_score >= 0.15:
                st.warning("Moderate risk — monitor and consider lifestyle changes.")
            else:
                st.success("Low risk — keep up healthy habits and monitor regularly.")

          
        else:
            # fallback for models without predict_proba
            pred = model.predict(X)[0]
            label = "Disease" if pred == 1 else "No disease"
            st.write("Prediction:", label)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer: tips
st.markdown("---")
st.markdown("---")