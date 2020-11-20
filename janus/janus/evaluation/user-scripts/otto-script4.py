# Source:
# https://www.kaggle.com/sachinsharma1123/otto-group-classification-acc-82from sklearn.pipeline import Pipeline

dataset = "otto"
metric = "neg_log_loss"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    # focused on SVC pipeline
    # to provide some diversity
    # (i.e. not just XGB pipelines)
    # probability=True needed for neg_log_loss
    clf = SVC(probability=True, random_state=0)
    p = Pipeline([("clf", clf)])
    return p
