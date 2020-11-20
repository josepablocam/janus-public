# Source:
# https://www.kaggle.com/sociopath00/forest-type-prediction-using-python

dataset = "forest-cover"
metric = "balanced_accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier(
        bootstrap=True,
        class_weight='balanced',
        criterion='gini',
        max_depth=None,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=300,
        n_jobs=1,
        oob_score=False,
        random_state=42,
        verbose=0,
        warm_start=False)
    p = Pipeline([("clf", clf)])
    return p
