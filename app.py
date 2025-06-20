import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
# --- Model and Preprocessor Loading ---
try:
    dis_model = joblib.load("disintegration_model.pkl")
except Exception as e:
    st.error(f"Error loading disintegration model: {e}")
    st.stop()

try:
    diss_model = load_model("dissolution_model.h5")
except Exception as e:
    st.error(f"Error loading dissolution model: {e}")
    st.stop()

try:
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")
    st.stop()

# --- UI ---

st.title("🧪 FormuPredict: AI for Disintegration & Dissolution")
binder = st.selectbox("Binder", ["PVP", "PEG"])
dis = st.selectbox("Disintegrant", ["Starch", "SSG", "Crosspovidone"])
ratio = st.number_input("Binder:Disintegrant Ratio", 0.5, 5.0, 2.0)
force = st.selectbox("Compression Force", ["Low", "Medium", "High"])
ph = st.slider("pH", 1.0, 9.0, 6.8)
hardness = st.slider("Tablet Hardness (kg/cm²)", 1.0, 10.0, 6.0)

input_df = pd.DataFrame([[binder, dis, ratio, force, ph, hardness]],
                        columns=['Binder', 'Disintegrant', 'Ratio_Binder_Dis', 'Compression_Force', 'pH', 'Hardness'])

# --- Prediction ---

# Disintegration time (scikit-learn)
try:
    dis_time = dis_model.predict(input_df)[0]
except Exception as e:
    st.error(f"Error predicting disintegration time: {e}")
    st.stop()

# Dissolution (Neural Network)
try:
    X_transformed = preprocessor.transform(input_df)
    diss_pred = diss_model.predict(X_transformed)[0]
except Exception as e:
    st.error(f"Error predicting dissolution: {e}")
    st.stop()

# --- Output ---

st.subheader("Prediction Results")
st.write(f"🕐 Disintegration Time: **{int(dis_time)} seconds**")

# Check output shape for dissolution prediction
if len(diss_pred) != 3:
    st.error("Dissolution model output shape is incorrect. Expected 3 values for 2h, 4h, 6h.")
else:
    chart_df = pd.DataFrame({
        "Dissolution %": diss_pred
    }, index=[2, 4, 6])
    st.line_chart(chart_df)
