import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    best_rf_model = pickle.load(f)

# Define the feature columns
feature_cols = ['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO']

# Function to preprocess input data
def preprocess_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[feature_cols])
    return data_scaled

# Streamlit app
def app():
    st.title("Urine Flow Classification")

    # Collect input data
    input_data = {}
    for col in feature_cols:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0, step=0.1)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    input_scaled = preprocess_data(input_df)

    # Reshape input data for prediction
    input_reshaped = input_scaled.reshape(1, -1)

    # Make prediction
    prediction = best_rf_model.predict(input_reshaped)
    predicted_class = prediction[0]

    # Display prediction
    st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    app()
