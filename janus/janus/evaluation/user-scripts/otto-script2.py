# Source:
# https://www.kaggle.com/abhishek/beating-the-benchmark-v2-0

dataset = "otto"
metric = "neg_log_loss"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(
        n_estimators=100,
        max_features=50,
        random_state=0,
    )
    p = Pipeline([("clf", clf)])
    return p
