import os
import traceback
import random

import openml
import numpy as np
import pandas as pd
import timeit
import tqdm
from stopit import threading_timeoutable, TimeoutException


def get_openml_dataset(dataset,
                       expand_categorical=True,
                       replace_nan_and_inf=None):
    print("Fetching {} from OpenML".format(dataset))
    dataset_obj = openml.datasets.get_dataset(dataset, download_data=True)
    X, y, _, _ = dataset_obj.get_data()
    if y is None:
        if dataset_obj.default_target_attribute is None:
            raise Exception("No target attribute for {}".format(dataset))
        y = X[dataset_obj.default_target_attribute]
        X = X.drop(columns=[dataset_obj.default_target_attribute])

    # expand categorical variables
    if expand_categorical:
        X = pd.get_dummies(X)

    if replace_nan_and_inf is not None:
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(replace_nan_and_inf)

    # return arrays only
    X = X.values
    y = y.values
    if y.dtype.name == "category":
        y = y.astype("str")
    return X, y


def get_local_dataset(name, expand_categorical=True):
    dataset_folder = os.environ.get(
        "DATASETS_FOLDER",
        os.path.join(os.path.dirname(__file__), "../../datasets/"))
    dataset_path = os.path.join(dataset_folder, name + ".tsv")
    if os.path.exists(dataset_path):
        dataset = pd.read_csv(dataset_path, sep='\t')
    else:
        dataset_path = os.path.join(dataset_folder, name + ".csv")
        dataset = pd.read_csv(dataset_path)
    X_df = dataset.drop(columns=["target"])
    # expand categorical variables
    if expand_categorical:
        X_df = pd.get_dummies(X_df)
    X = X_df.values
    y = dataset['target'].values
    return X, y


def get_dataset(dataset, expand_categorical=True, replace_nan_and_inf=None):
    try:
        return get_local_dataset(dataset, expand_categorical)
    except:
        return get_openml_dataset(dataset, expand_categorical,
                                  replace_nan_and_inf)


def read_and_concat(paths):
    acc = []
    for p in tqdm.tqdm(paths):
        if p.endswith("pkl"):
            df = pd.read_pickle(p)
        elif p.endswith("csv"):
            df = pd.read_csv(p)
        else:
            raise ValueError("Unknown file type:{}".format(p))
        acc.append(df)

    comb_df = pd.concat(acc, axis=0).reset_index(drop=True)
    return comb_df


@threading_timeoutable(default="Timeout", timeout_param="timeout")
def safetimeit(closure, number=1):
    try:
        return timeit.timeit(closure, number=number)
    except Exception as err:  # NOQA
        return np.nan


@threading_timeoutable(default="Timeout", timeout_param="timeout")
def call_with_timeout(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except TimeoutException:
        # needs to be different type, otherwise decorator swallows exception
        # see stopit documentation for details
        raise TimeoutError


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_exception_name(err):
    try:
        return err.__name__
    except AttributeError:
        return str(type(err))


def get_exception_tb(err):
    try:
        tb_msg = "".join(
            traceback.TracebackException.from_exception(err).format())
        return tb_msg
    except AttributeError:
        return str(err)
