import pickle


def predict_(data):
    with open('rf_model.pkl' , 'rb') as f:
    clf = pickle.load(f)  
    return clf.predict(data)
