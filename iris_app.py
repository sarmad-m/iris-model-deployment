# import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


clf = pickle.load(open('rf_model.pkl','rb'))


## ******************* THE web APP ************************
# title and description:
st.title('Classifying Iris Flowers')
st.markdown('Toy model to play with the iris flowers dataset and classify the three Species into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Input Features")
Pr = float(st.number_input("Pr"))
Frate = float(st.number_input("Frate"))
Favrg = float(st.number_input("Favrg"))
Time = float(st.number_input("Time"))
Vtotal = float(st.number_input("Vtotal"))
Fmax = float(st.number_input("Fmax"))
Tmax = float(st.number_input("Tmax"))
SNO = float(st.number_input("SNO"))

# Combine input features into a 1D NumPy array
input_features = np.array([Pr, Frate, Favrg, Time, Vtotal, Fmax, Tmax, SNO])

# Reshape the array to be a 2D array with a single row
input_features = input_features.reshape(1, -1)

# Scale the input features using StandardScaler
sc = StandardScaler()
scaled_features = sc.fit_transform(input_features)

# prediction button
if st.button("Predict"):
    result = clf.predict(scaled_features)
    st.text("Predicted Class:", result[0])


st.text('')
st.text('')
st.markdown(
    '`Initial code was developed by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code modification and update by:` [Mohamed Alie](https://github.com/Kmohamedalie/iris-streamlit/tree/main)')
