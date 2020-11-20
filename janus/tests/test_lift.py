#!/usr/bin/env python3
import pytest

import numpy as np
import sklearn.linear_model
import sklearn.datasets
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.lift.extract_pipeline import PipelineLifter
from tests.utils import PipelineGenerator

import copy
import sys
sys.path.append(".")


def test_lift1():
    src = """
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

    import numpy as np
    X = np.random.random((10, 10))
    y = np.random.random(10)
    y = y > 0.5

    p.fit(X, y)
    p.predict(X)
    """
    result = PipelineLifter(src)
    assert not result.failed
    assert len(result.pipelines) == 1

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
    expected = Pipeline([('scaler', StandardScaler()), ('logit', logit)])

    assert pt.md5(expected) == pt.md5(result.pipelines[0])


def test_lift2():
    # same pipeline but now not using the Pipeline construct in the
    # source
    src = """
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

    import numpy as np
    X = np.random.random((10, 10))
    y = np.random.random(10)
    y = y > 0.5

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    logit.fit(X_scaled, y)
    logit.predict(X_scaled)
    """
    result = PipelineLifter(src)
    assert not result.failed
    assert len(result.pipelines) == 1

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
    expected = Pipeline([('scaler', StandardScaler()), ('logit', logit)])

    assert pt.md5(expected) == pt.md5(result.pipelines[0])


def test_lift3():
    # https://www.kaggle.com/vsmolyakov/svm-classifier
    # with some mods --> remove deprecated stale/code from sklearn
    # in source (from older version, incompatible)
    src = """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv as csv

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler


    # subbed out data here...
    X_train = np.random.random((100, 100))
    y_train = np.random.random(100) > 0.5


    svm_parameters = [{'kernel': ['rbf'], 'C': [1,10,100,1000]}]
    clf = GridSearchCV(SVC(), svm_parameters, cv=3, verbose=2)
    clf.fit(X_train, y_train)
    clf.best_params_

    C_opt = 10
    clf = SVC(C=C_opt, kernel='rbf')
    clf.fit(X_train, y_train)
    clf.n_support_

    X_test_data = np.random.random((100, 100))
    y_pred = clf.predict(X_test_data)
    """
    result = PipelineLifter(src)
    assert not result.failed
    assert len(result.pipelines) == 2

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    clf = SVC(C=10, kernel="rbf")
    expected1 = Pipeline([("clf", clf)])

    svm_parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), svm_parameters, cv=3, verbose=2)
    expected2 = Pipeline([("clf", clf)])

    expected = set([pt.md5(expected1), pt.md5(expected2)])
    got = set([pt.md5(p) for p in result.pipelines])
    assert got == expected


def test_lift4():
    # now no .predict call, but has a .fit call
    src = """
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

    import numpy as np
    X = np.random.random((10, 10))
    y = np.random.random(10)
    y = y > 0.5

    p.fit(X, y)
    """
    result = PipelineLifter(src)
    assert not result.failed
    assert len(result.pipelines) == 1

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
    expected = Pipeline([('scaler', StandardScaler()), ('logit', logit)])

    assert pt.md5(expected) == pt.md5(result.pipelines[0])
