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

# features sliders for the four plant features:
st.header("Plant Features")
st.header("Input Features")
Pr =  float(st.number_input("Pr"))
Frate = float(st.number_input("Frate"))
Favrg = float(st.number_input("Favrg"))
Time = float(st.number_input("Time"))
Vtotal = float(st.number_input("Vtotal"))
Fmax = float(st.number_input("Fmax"))
Tmax = float(st.number_input("Tmax"))
SNO = float(st.number_input("SNO"))

st.text('')

sc=StandardScaler()
columns_to_scale=['Pr','Frate','Favrg','Time','Vtotal','Fmax','Tmax',	'SNO']
cols=sc.fit_transform(train[columns_to_scale])

# prediction button
if st.button("Predicts"):
    result = clf.predict(cols)
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Initial code was developed by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code modification and update by:` [Mohamed Alie](https://github.com/Kmohamedalie/iris-streamlit/tree/main)')
