# Source:
# https://www.kaggle.com/samratp/machine-learning-with-ghouls-goblins-and-ghosts

dataset = "ghouls"
metric = "accuracy"


def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import (
        VotingClassifier,
        RandomForestClassifier,
        BaggingClassifier,
        GradientBoostingClassifier,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    bag = BaggingClassifier(max_samples=5, n_estimators=25, random_state=0)
    gb = GradientBoostingClassifier(
        learning_rate=0.1, max_depth=5, n_estimators=100, random_state=0)
    lr = LogisticRegression(
        penalty='l1', C=1, random_state=0, solver="liblinear")
    svc = SVC(
        C=10, degree=3, kernel='linear', probability=True, random_state=0)

    clf = VotingClassifier(
        estimators=[('rf', rf), ('bag', bag), ('gbc', gb), ('lr', lr),
                    ('svc', svc)],
        voting='hard',
    )
    p = Pipeline([('clf', clf)])
    return p
