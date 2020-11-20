import traceback

import sklearn.base


class RobustSearch(sklearn.base.BaseEstimator):
    def __init__(self, search_model, verbose=False):
        self.search_model = search_model
        self.verbose = verbose

    def fit(self, X, y):
        try:
            self.search_model.fit(X, y)
        except:
            if self.verbose:
                error_msg = traceback.format_exc()
                print(error_msg)

    def __getattr__(self, attr_name):
        return getattr(self.search_model, attr_name)
