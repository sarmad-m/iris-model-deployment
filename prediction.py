import pickle
from sklearn.externals import joblib
model = joblib.load('rf_model.pkl')

def predict_(data, model = model):   
    return model.predict(data)
