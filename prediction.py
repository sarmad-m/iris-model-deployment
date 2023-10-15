import pickle
import joblib
model = joblib.load('rf_model.pkl')
model = pickle.load('rf_model.pkl')


def predict_(data, model = model):   
    return model.predict(data)
