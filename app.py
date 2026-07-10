import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("ckd_random_forest.pkl")

st.title("🩺 Chronic Kidney Disease Prediction")

st.write("Enter patient details")

# Get feature names used during training
features = model.feature_names_in_

input_data = {}

# Create input fields automatically
for feature in features:
    
    if "Diabetes" in feature or "Hypertension" in feature:
        input_data[feature] = st.selectbox(
            feature,
            [0, 1]
        )

    elif "Gender" in feature or "Sex" in feature:
        input_data[feature] = st.selectbox(
            feature,
            [0, 1]
        )

    else:
        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )


# Convert to dataframe
input_df = pd.DataFrame([input_data])


# Prediction
if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Chronic Kidney Disease Detected")
    else:
        st.success("✅ No Chronic Kidney Disease Detected")