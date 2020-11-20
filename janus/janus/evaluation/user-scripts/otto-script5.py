# Source:
# https://www.kaggle.com/zmey56/using-t-sne-and-pca-for-otto-group-product

dataset = "otto"
metric = "neg_log_loss"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=2, random_state=0)
    clf = LogisticRegression(
        C=1000.0,
        dual=False,
        fit_intercept=True,
        max_iter=100,
        penalty="l2",
        solver="liblinear",
        tol=0.0001,
        random_state=0,
    )
    p = Pipeline([("pca", pca), ("clf", clf)])
    return p
