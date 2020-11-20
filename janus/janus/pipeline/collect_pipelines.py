#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import pickle
import random
import timeit
import time
import traceback
import sys

import numpy as np
import pandas as pd
import tpot
import tpot.config
import tqdm
import sklearn.base
import sklearn.pipeline
import sklearn.metrics
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from janus.pipeline import pipeline_to_tree as pt
from janus import utils
from janus.search import random_search
from janus.search.utils import RobustSearch


def get_robust_tpot(
        max_time_mins=5,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=1,
        verbose=False,
):
    clf = RobustSearch(
        search_model=tpot.TPOTClassifier(
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            max_time_mins=max_time_mins,
            random_state=random_state,
            verbosity=2,
            disable_update_check=True,
        ),
        verbose=verbose,
    )
    return clf


def get_robust_random_search(
        max_time_mins=5,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=1,
        verbose=False,
):
    clf = RobustSearch(
        search_model=random_search.RandomSearch(
            tpot.config.classifier_config_dict,
            max_depth=4,
            max_time_mins_per_pipeline=1,
            max_retries=100,
            max_time_mins=max_time_mins,
            cv=cv,
            scoring=scoring,
            random_state=random_state),
        verbose=verbose,
    )
    return clf


def collect_exceptions(individuals):
    records = []
    for obj_key, entry in tqdm.tqdm(individuals.items()):
        obj = entry.get("pipeline_obj", None)
        if obj is None:
            continue
        obj_graph = pt.to_tree(obj)
        record = {
            "obj": sklearn.base.clone(obj),
            "obj_key": obj_key,
            "obj_graph": obj_graph,
            "failed": True,
            # "exception_type": str(type(entry["exception"])),
            # "exception_str": str(entry["exception"]),
        }
        records.append(record)
    return records


def collect_time_and_perf(
        individuals,
        X_train,
        y_train,
        X_test,
        y_test,
        scoring,
        max_time_mins,
        include_prediction_time=False,
):
    records = []
    for obj_key, entry in tqdm.tqdm(individuals.items()):
        obj = entry.get("pipeline_obj", None)
        if obj is None:
            # bad entry, don't care
            continue

        # fit on entire X_train
        train_time = np.nan
        try:
            train_time = utils.safetimeit(
                lambda: obj.fit(X_train, y_train),
                timeout=(max_time_mins * 60),
            )
        except utils.TimeoutException:
            print("Timed out")

        if np.isnan(train_time):
            external_score = np.nan
        else:
            # performance on test set
            scorer = sklearn.metrics.SCORERS[scoring]
            try:
                external_score = scorer(obj, X_test, y_test)
            except:
                continue

        # graph representation
        try:
            obj_graph = pt.to_tree(obj)
        except:
            continue

        record = {
            "obj": sklearn.base.clone(obj),
            "obj_key": obj_key,
            "obj_graph": obj_graph,
            "train_time": train_time,
            "external_score": external_score,
            "failed": False,
            "timestamp": entry["timestamp"],
        }
        records.append(record)
    return records


def run_dataset(
        dataset,
        search,
        config=None,
        max_time_mins=5,
        hold_out_size=0.5,
        test_size=0.2,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=None,
        verbose=False,
        include_prediction_time=False,
):
    X, y = utils.get_dataset(dataset)

    if search == "tpot":
        # running search with tpot
        model = get_robust_tpot(
            max_time_mins=max_time_mins,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
    elif search == "random":
        model = get_robust_random_search(
            max_time_mins=max_time_mins,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise Exception("Unknown search: " + search)

    idx = np.arange(0, X.shape[0])
    X_use, _, y_use, _, idx_search, idx_hold_out = train_test_split(
        X,
        y,
        idx,
        test_size=hold_out_size,
        stratify=y,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_use,
        y_use,
        test_size=test_size,
        stratify=y_use,
    )

    print("Running search")
    model.fit(X_train, y_train)
    print("Done running search")

    records = []
    #print("Accumulating failed pipelines")
    #failures = collect_exceptions(model.search_model.failed_individuals_)
    #records.extend(failures)

    print("Training/evaluating successfull candidates generated")
    successes = collect_time_and_perf(
        model.search_model.evaluated_individuals_,
        X_train,
        y_train,
        X_test,
        y_test,
        scoring,
        max_time_mins,
        include_prediction_time=include_prediction_time,
    )
    records.extend(successes)

    results_df = pd.DataFrame(records)
    results_df["dataset"] = dataset
    results_df["search"] = search
    return results_df, idx_search, idx_hold_out


def to_df(acc, args):
    acc_df = pd.concat(acc, axis=0)
    acc_df["random_state"] = args.random_state
    return acc_df


def get_args():
    parser = ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Name of datasets to run",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search strategy",
        choices=[
            "tpot",
            "random",
        ])
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring function",
        default="f1_macro",
    )
    parser.add_argument(
        "--max_time_mins",
        type=int,
        help="Time budget for each outer cv iteration",
        default=60,
    )
    parser.add_argument(
        "--hold_out_size",
        type=float,
        help="Part of data to completely hold out for downstream evaluation",
        default=0.5,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="Test split fraction (to evaluate searched pipelines)",
        default=0.2,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of cores to use",
        default=1,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Seed for RNG",
        default=42,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output file exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose errors and printouts",
    )
    parser.add_argument(
        "--include_prediction_time",
        action="store_true",
        help="Add in prediction time on training data as well to time",
    )
    return parser.parse_args()


def main():
    args = get_args()
    acc = []
    if args.random_state:
        # adding more set seeds....something deep down
        # in tpot/sklearn not actually taking the random seed otherwise
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    if args.output is not None:
        dir_path = os.path.dirname(args.output)
        if len(dir_path) > 0:
            os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(args.output) and not args.force:
        print("{} exists, use --force to re-run".format(args.output))
        return 0

    for dataset in tqdm.tqdm(args.dataset):
        print("Running dataset", dataset)
        # we searched for models on idx_search
        # and *did* not touch idx_test
        results, idx_search, idx_hold_out = run_dataset(
            dataset,
            search=args.search,
            hold_out_size=args.hold_out_size,
            test_size=args.test_size,
            max_time_mins=args.max_time_mins,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=args.verbose,
            include_prediction_time=args.include_prediction_time,
        )
        acc.append(results)
        temp_df = to_df(acc, args)
        temp_df.to_pickle(args.output + "-tmp")
        with open(args.output + "-idx-search", "wb") as fout:
            pickle.dump(idx_search, fout)
        with open(args.output + "-idx-hold-out", "wb") as fout:
            pickle.dump(idx_hold_out, fout)

    final_df = to_df(acc, args)
    final_df.to_pickle(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        args = get_args()
        err_path = args.output + "-error"
        with open(err_path, "wb") as fout:
            pickle.dump(err, fout)

        detailed_msg = traceback.format_exc()
        tb_path = args.output + "-tb"
        with open(tb_path, "w") as fout:
            fout.write(detailed_msg)
            fout.write("\n")

        failed_args_path = args.output + "-args"
        with open(failed_args_path, "wb") as fout:
            pickle.dump(args, fout)

        import pdb
        pdb.post_mortem()
        sys.exit(1)
