import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the pre-trained model
model = joblib.load("rf_model.pkl")

# Streamlit app
st.title("Urine Flow Classification")

# Sidebar for input
st.sidebar.header("Input Parameters")
input_params = {}
columns_to_scale = ['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO']
for col in columns_to_scale:
    input_params[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)

# Submit button
if st.sidebar.button("Predict"):
    # Preprocessing function to handle missing values
    def preprocess_data(data):
        data = data.fillna(0)  # Fill missing values with zeros
        return data

    # Prediction
    input_data = pd.DataFrame([input_params])
    input_data = preprocess_data(input_data)
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.write("## Prediction")
    st.write(f"The predicted class is: {prediction}")
