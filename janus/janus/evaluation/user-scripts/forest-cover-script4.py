# Source:
# https://www.kaggle.com/sharmasanthosh/exploratory-study-of-ml-algorithms
# https://www.kaggle.com/sharmasanthosh/exploratory-study-on-feature-selection

dataset = "forest-cover"
metric = "balanced_accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFE
    from sklearn.neighbors import KNeighborsClassifier

    # a lot of pipelines in this one
    # so we focus on one that is different
    # from those that we have tried so far
    # (i.e. not just another RF)
    scaler = StandardScaler()
    feat_select = RFE(LogisticRegression(random_state=0, n_jobs=1))
    knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
    p = Pipeline([
        ('scaler', scaler),
        ('feat', feat_select),
        ('clf', knn),
    ])
    return p
