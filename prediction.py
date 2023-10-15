import pickle


def predict_(data):
    clf = pickle.load('rf_model.pkl')
    return clf.predict(data)
