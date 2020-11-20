# Source:
# https://www.kaggle.com/yoyocm/let-s-explore-and-classify-monsters

dataset = "ghouls"
metric = "accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    clf = LogisticRegression(penalty='l2', C=1000000, random_state=0)
    p = Pipeline([('clf', clf)])
    return p
