# import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

clf = pickle.load(open('rf_model.pkl', 'rb'))

# Title and description
st.title('Classifying Urine Flowmeter')
st.header("Input Features")

# Input features
Pr =  st.number_input("Pr")
Frate = st.number_input("Frate")
Favrg = st.number_input("Favrg")
Time = st.number_input("Time")
Vtotal = st.number_input("Vtotal")
Fmax = st.number_input("Fmax")
Tmax = st.number_input("Tmax")
SNO = st.number_input("SNO")

# Combine input features into a 2D array
input_features = np.array([[Pr, Frate, Favrg, Time, Vtotal, Fmax, Tmax, SNO]])


# Prediction button
if st.button("Predict"):
    result = clf.predict(input_features)
    st.text("Predicted Class:", result[0])
