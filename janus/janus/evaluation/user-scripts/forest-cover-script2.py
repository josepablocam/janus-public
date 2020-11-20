# Source:
# https://www.kaggle.com/triskelion/first-try-with-random-forests

dataset = "forest-cover"
metric = "balanced_accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(
        n_estimators=200, n_jobs=1, random_state=0)
    p = Pipeline([("clf", clf)])
    return p
