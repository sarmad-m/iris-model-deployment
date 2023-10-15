import pickle


def predict(data):
    clf = pickle.load(open('rf_model.sav', 'rb'))
    return clf.predict(data)
