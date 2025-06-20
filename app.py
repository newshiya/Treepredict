import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load models
dis_model = joblib.load("disintegration_model.pkl")
diss_model = load_model("dissolution_model.h5")

# UI
st.title("🧪 FormuPredict: AI for Disintegration & Dissolution")
binder = st.selectbox("Binder", ["PVP", "PEG"])
dis = st.selectbox("Disintegrant", ["Starch", "SSG", "Crosspovidone"])
ratio = st.number_input("Binder:Disintegrant Ratio", 0.5, 5.0, 2.0)
force = st.selectbox("Compression Force", ["Low", "Medium", "High"])
ph = st.slider("pH", 1.0, 9.0, 6.8)
hardness = st.slider("Tablet Hardness (kg/cm²)", 1.0, 10.0, 6.0)

input_df = pd.DataFrame([[binder, dis, ratio, force, ph, hardness]],
                        columns=['Binder', 'Disintegrant', 'Ratio_Binder_Dis', 'Compression_Force', 'pH', 'Hardness'])

# Predict disintegration time
dis_time = dis_model.predict(input_df)[0]

# Encode for NN model
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Binder', 'Disintegrant', 'Compression_Force'])
], remainder='passthrough')
X_transformed = preprocessor.fit_transform(input_df)

# Predict dissolution
diss_pred = diss_model.predict(X_transformed)[0]

# Output
st.subheader("Prediction Results")
st.write(f"🕐 Disintegration Time: **{int(dis_time)} seconds**")
st.line_chart({
    "Dissolution %": diss_pred,
    "Time (h)": [2, 4, 6]
})