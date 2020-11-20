# Source:
# https://www.kaggle.com/mikhailg0/monsters-classification-solution

dataset = "ghouls"
metric = "accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.neighbors import KNeighborsClassifier

    scaler = Normalizer()
    clf = KNeighborsClassifier(n_neighbors=5)
    p = Pipeline([('scaler', scaler), ('clf', clf)])
    return p
