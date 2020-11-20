# Source:
# https://www.kaggle.com/lilyelizabethjohn/standardization-using-standardscaler

dataset = "ghouls"
metric = "accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    clf = LogisticRegression(random_state=0)
    p = Pipeline([('scaler', scaler), ('clf', clf)])
    return p
