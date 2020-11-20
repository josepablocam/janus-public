# Source:
# https://www.kaggle.com/kashnitsky/topic-10-practice-with-logit-rf-and-lightgbm

dataset = "forest-cover"
metric = "balanced_accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(
        C=1,
        solver='lbfgs',
        max_iter=500,
        random_state=17,
        n_jobs=1,
        multi_class='multinomial')
    p = Pipeline([('scaler', StandardScaler()), ('logit', logit)])
    return p
