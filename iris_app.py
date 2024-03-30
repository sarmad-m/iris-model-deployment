import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("Urineflowmeterdata.csv")

# Preprocessing
features = train.loc[:, 'Pr':'SNO']
scaler = StandardScaler()
columns_to_scale = ['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO']
features[columns_to_scale] = scaler.fit_transform(features[columns_to_scale])
classes = train['class']
le = LabelEncoder()
classes = le.fit_transform(classes)

# Oversampling
def oversampling(X_train, y_train):
    sm = SMOTE(random_state=40, sampling_strategy=dict)
    x_smote_train, y_smote_train = sm.fit_resample(X_train, y_train)
    return x_smote_train, y_smote_train

x_smote_train, y_smote_train = oversampling(features, classes)

# Model training
X_train, X_test, y_train, y_test = train_test_split(x_smote_train, y_smote_train, test_size=0.2, random_state=40)
rf_classifier = RandomForestClassifier()
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2], 'min_samples_leaf': [1]}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Streamlit app
st.title("Urine Flow Classification")

# Sidebar for input
st.sidebar.header("Input Parameters")
input_params = {}
for col in columns_to_scale:
    input_params[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)

# Prediction
input_data = pd.DataFrame([input_params])
input_data_scaled = scaler.transform(input_data)
prediction = best_rf_model.predict(input_data_scaled)
prediction_class = le.inverse_transform(prediction)[0]

# Display prediction
st.write("## Prediction")
st.write(f"The predicted class is: {prediction_class}")
