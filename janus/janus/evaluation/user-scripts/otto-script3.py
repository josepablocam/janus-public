# Source:
# https://www.kaggle.com/ankitdatascience/random-and-bayes-search-hyp-optimization-gpu

dataset = "otto"
metric = "neg_log_loss"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    # Ignoring gpu_hist (no access to GPU)
    # Ignoring n_thread=6 (single threaded)
    clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        colsample_bytree=0.8,
        learning_rate=0.1,
        criterion="entropy",
        random_state=0,
        )
    p = Pipeline([("clf", clf)])
    return p
