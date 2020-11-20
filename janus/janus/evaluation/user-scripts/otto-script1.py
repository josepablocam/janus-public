# Source:
# https://www.kaggle.com/ankitdatascience/xgboost-for-otto-dataset

dataset = "otto"
metric = "neg_log_loss"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.2,
        colsample_bytree=0.7,
        random_state=0,
    )
    p = Pipeline([("clf", clf)])
    return p
