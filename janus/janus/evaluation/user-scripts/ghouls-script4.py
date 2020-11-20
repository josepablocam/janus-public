# Source:
# https://www.kaggle.com/anki08/classifying-ghouls-goblins-and-ghosts-using-svm

dataset = "ghouls"
metric = "accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    clf = SVC(C=5, gamma=0.5, random_state=0)
    p = Pipeline([('clf', clf)])
    return p
