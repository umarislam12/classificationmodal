import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
# print(np.__version__)
# Set app title
st.title("Water Potability Prediction")


# Load the saved model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load('water_potability_model.joblib')
    return model


@st.cache_data
def load_features():
    with open('feature_columns.json', 'r') as f:
        features = json.load(f)
    return features


model = load_model()
feature_columns = load_features()

# Create input widgets in sidebar
st.sidebar.header("User Input Parameters")


def user_input_features():
    inputs = {}

    inputs['ph'] = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
    inputs['Hardness'] = st.sidebar.slider("Hardness (mg/L)", 0.0, 400.0, 200.0)
    inputs['Solids'] = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, 20000.0)
    inputs['Chloramines'] = st.sidebar.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
    inputs['Sulfate'] = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 300.0)
    inputs['Conductivity'] = st.sidebar.slider("Conductivity (μS/cm)", 0.0, 1000.0, 500.0)
    inputs['Organic_carbon'] = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 15.0)
    inputs['Trihalomethanes'] = st.sidebar.slider("Trihalomethanes (μg/L)", 0.0, 150.0, 80.0)
    inputs['Turbidity'] = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 5.0)

    # Convert to DataFrame with correct column order
    features = pd.DataFrame([inputs], columns=feature_columns)
    return features


# Display user input
st.header("Water Quality Parameters")
st.write("Adjust the sliders in the sidebar and see the prediction")

df = user_input_features()
st.write(df)

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader("Prediction")
potability = np.array(['Not Potable', 'Potable'])
st.success(potability[prediction][0])

st.subheader("Prediction Probability")
st.write(f"Not Potable: {prediction_proba[0][0] * 100:.2f}%")
st.write(f"Potable: {prediction_proba[0][1] * 100:.2f}%")

# Optional: Add some explanation
st.markdown("""
### About This App
This app predicts water potability based on 9 different water quality parameters using a Logistic Regression model.
- **Not Potable (0)**: Water is not safe for drinking
- **Potable (1)**: Water is safe for drinking
""")