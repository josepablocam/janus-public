# Source:
# https://www.kaggle.com/vsmolyakov/svm-classifier

dataset = "forest-cover"
metric = "balanced_accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    scaler = StandardScaler()
    clf = SVC(C=10, kernel='rbf', probability=True, random_state=0)
    p = Pipeline([
        ('scaler', scaler),
        ('clf', clf),
    ])
    return p
