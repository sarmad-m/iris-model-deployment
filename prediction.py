import pickle


def predict_(data):
    clf = pickle.load(open('rf_model.sav', 'rb'))
    return clf.predict(data)
