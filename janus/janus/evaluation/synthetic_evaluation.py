#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.local_rules import (
    HyperparamUpdate,
    HyperparamRemove,
    ComponentUpdate,
    ComponentRemove,
    ComponentInsert,
    RuleCorpus,
)
from janus.repair.tree_enumerator import (
    get_tree_enumerator,
)
from janus.repair.rule_sampler import (
    get_rule_sampler,
)
from janus.repair.repairer import PipelineRepairer
from janus import utils
from janus import mp_utils

if os.environ.get("DEBUG", None) is None:
    DEFAULT_TIMEOUT_EVAL = 60
else:
    DEFAULT_TIMEOUT_EVAL = int(1e6)


def run_single_tree(
    X_search,
    y_search,
    X_test,
    y_test,
    test_pipeline_tree,
    enumerator,
    bound_num_repaired_pipelines,
    dev_cv=3,
    bound_k=3,
    cv=5,
    scoring="f1_macro",
    random_state=42,
):
    repairer = PipelineRepairer(enumerator)

    results_summary = []
    orig_info = {
        "type": "orig",
        "graph": test_pipeline_tree,
    }

    orig_compiled = pt.to_pipeline(test_pipeline_tree)


    # TODO: this should be a param
    # should be about 5% of dataset, since search is 50%
    num_obs_search = int(X_search.shape[0] * 0.1)
    assert num_obs_search >= 1
    if isinstance(X_search, pd.DataFrame):
        X_search = X_search.values
    if isinstance(y_search, (pd.DataFrame, pd.Series)):
        y_search = y_search.values

    X_search = X_search[:num_obs_search]
    y_search = y_search[:num_obs_search]

    utils.set_seed(random_state)
    repaired = repairer.repair(
        orig_compiled,
        X_search,
        y_search,
        bound_k=bound_k,
        bound_num_repairs=bound_num_repaired_pipelines,
        scoring=scoring,
        cv=dev_cv,
        random_state=random_state,
        verbosity=1,
    )

    try:
        print("Evaluate original")
        utils.set_seed(random_state)
        orig_results = mp_utils.run(
            DEFAULT_TIMEOUT_EVAL,
            cross_validate,
            orig_compiled,
            X_test,
            y_test,
            cv=StratifiedKFold(
                cv,
                random_state=random_state,
                shuffle=True,
            ),
            scoring=scoring,
            return_estimator=True,
            return_train_score=True,
        )
        orig_info["test_scores"] = orig_results["test_score"]
        orig_info["mean_test_score"] = np.mean(orig_results["test_score"])
        orig_info["failed"] = False
        orig_info["timedout"] = False
    except mp_utils.TimeoutError:
        print("Timedout on original pipeline")
        orig_info["failed"] = True
        orig_info["timedout"] = True
        orig_info["test_scores"] = []
        orig_info["mean_test_score"] = np.nan
    except Exception as err:
        print("Failed to run original pipeline")
        print(err)
        orig_info["failed"] = True
        orig_info["timedout"] = False
        orig_info["test_scores"] = []
        orig_info["mean_test_score"] = np.nan

    if repaired is None:
        print("No repair found")
        orig_info["no_repaired_candidates"] = True
        results_summary.append(orig_info)
        return pd.DataFrame(results_summary)
    else:
        orig_info["no_repaired_candidates"] = False

    results_summary.append(orig_info)

    repair_info = {
        "type": "repair",
        "graph": pt.to_tree(repaired),
        "no_repaired_candidates": False,
    }
    try:
        print("Evaluate repaired")
        utils.set_seed(random_state)
        repaired_results = mp_utils.run(
            DEFAULT_TIMEOUT_EVAL,
            cross_validate,
            repaired,
            X_test,
            y_test,
            cv=StratifiedKFold(
                cv,
                random_state=random_state,
                shuffle=True,
            ),
            scoring=scoring,
            return_estimator=True,
            return_train_score=True,
        )
        repair_info["test_scores"] = repaired_results["test_score"]
        repair_info["mean_test_score"] = np.mean(
            repaired_results["test_score"]
        )
        repair_info["failed"] = False
        repair_info["timedout"] = False
    except mp_utils.TimeoutError:
        print("Timedout on repair pipeline")
        orig_info["failed"] = True
        orig_info["timedout"] = True
        orig_info["test_scores"] = []
        orig_info["mean_test_score"] = np.nan
    except Exception as err:
        print("Failed to run repaired pipeline")
        print(err)
        repair_info["test_scores"] = []
        repair_info["mean_test_score"] = np.nan
        repair_info["failed"] = True
        repair_info["timedout"] = False

    repair_info["repairer_statistics"] = repairer.statistics
    orig_info["repairer_statistics"] = None
    results_summary.append(repair_info)
    return pd.DataFrame(results_summary)


def run_evaluation(
    dataset,
    timestamped_test_pipeline_trees,
    enumerator,
    bound_num_repaired_pipelines,
    idx_search,
    dev_cv=3,
    bound_k=3,
    cv=5,
    scoring="f1_macro",
    random_state=42,
):
    results = []

    X, y = dataset
    used_for_search = set(idx_search)
    idx_test = [
        i for i in np.arange(0, X.shape[0]) if i not in used_for_search
    ]

    X_search, y_search = X[idx_search], y[idx_search]
    X_test, y_test = X[idx_test], y[idx_test]

    for ix, elem in tqdm.tqdm(list(
            enumerate(timestamped_test_pipeline_trees))):
        if isinstance(elem, tuple):
            tree, timestamp = elem
        else:
            tree = elem
            timestamp = np.nan

        tree_results = run_single_tree(
            X_search,
            y_search,
            X_test,
            y_test,
            tree,
            enumerator,
            bound_num_repaired_pipelines,
            dev_cv=dev_cv,
            bound_k=bound_k,
            cv=cv,
            scoring=scoring,
            random_state=random_state + ix,
        )
        tree_results["id"] = ix
        tree_results["scoring"] = scoring
        tree_results["cv"] = cv
        tree_results["bound_k"] = bound_k
        tree_results["bound_num_repaired_pipelines"
                     ] = bound_num_repaired_pipelines
        tree_results["random_state"] = random_state + ix
        tree_results["timestamp"] = timestamp
        results.append(tree_results)
    df_results = pd.concat(results, axis=0)
    return df_results


def get_args():
    parser = ArgumentParser(
        description="Evaluate repairs on a set of existing pipelines"
    )
    parser.add_argument(
        "--rules",
        type=str,
        nargs="+",
        help="Local rule files",
    )
    parser.add_argument(
        "--rule_strategy",
        type=str,
        help="Rule sampling strategy",
    )
    parser.add_argument(
        "--enumeration_strategy",
        type=str,
        help="Tree enumeration strategy",
    )
    parser.add_argument(
        "--predefined_strategy",
        choices=["weighted-transducer", "rf-transducer", "random-mutation"],
        help="Predefined strategies",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="File with pipelines on which to evaluate",
    )
    parser.add_argument(
        "--num_test_pipelines",
        type=int,
        help="Number of pipelines to repair (downsamples if less than total)",
    )
    parser.add_argument(
        "--bound_num_repaired_pipelines",
        type=int,
        help="Number of new (repaired) pipelines to eval",
        default=10,
    )
    parser.add_argument(
        "--bound_k",
        type=int,
        help="Enumerator-specific bound k (limiting number of rules applied)",
        default=3,
    )
    parser.add_argument(
        "--idx_search",
        type=str,
        help="Path to indices used for the original search of pipelines",
    )
    parser.add_argument(
        "--dev_cv",
        type=int,
        help="Number of CV folds for repair validation",
        default=3,
    )
    parser.add_argument(
        "--cv",
        type=int,
        help="Number of CV folds for evaluation",
        default=5,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring function",
        default="f1_macro",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to dump results",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="RNG seed",
        default=42,
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.random_state is not None:
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    rules = []
    if args.predefined_strategy is not None:
        assert args.rule_strategy is None
        assert args.enumeration_strategy is None
    if args.predefined_strategy == "weighted-transducer":
        args.rule_strategy = "weighted"
        args.enumeration_strategy = "beam"
    elif args.predefined_strategy == "rf-transducer":
        args.rule_strategy = "predictive"
        args.enumeration_strategy = "beam"
    elif args.predefined_strategy == "random-mutation":
        args.rule_strategy = "mutation"
        args.enumeration_strategy = "beam"
        args.bound_k = args.bound_num_repaired_pipelines
    else:
        raise Exception(
            "Unknown predefined_strategy: " + args.predefined_strategy
        )
    rules = []
    if args.rule_strategy != "mutation":
        for p in args.rules:
            with open(p, "rb") as fin:
                rule_corpus = pickle.load(fin)
                rules.extend(rule_corpus.rules)
    rule_sampler = get_rule_sampler(
        args.rule_strategy,
        rules,
        args.random_state,
    )
    enumerator = get_tree_enumerator(
        args.enumeration_strategy,
        rule_sampler,
        force_apply=(args.rule_strategy == "mutation")
    )

    df_test_data = pd.read_pickle(args.test)
    # only focus on pipelines that worked before
    df_test_data = df_test_data[~df_test_data["failed"]]
    df_test_data = df_test_data.reset_index(drop=True)

    if df_test_data.shape[0] > args.num_test_pipelines:
        df_test_data = df_test_data.sample(
            n=args.num_test_pipelines,
            replace=False,
            random_state=args.random_state,
        )

    df_test_data = df_test_data.sort_values("timestamp", ascending=True)
    # graph representation of the pipeline
    test_trees = df_test_data["obj_graph"].values
    timestamps = df_test_data["timestamp"].values
    timestamped_test_trees = zip(test_trees, timestamps)

    test_dataset_name = df_test_data.iloc[0]["dataset"]
    test_dataset = utils.get_dataset(
        test_dataset_name,
        replace_nan_and_inf=0.0,
    )

    with open(args.idx_search, "rb") as fin:
        idx_search = pickle.load(fin)

    df_results = run_evaluation(
        test_dataset,
        timestamped_test_trees,
        enumerator,
        args.bound_num_repaired_pipelines,
        idx_search=idx_search,
        dev_cv=args.dev_cv,
        bound_k=args.bound_k,
        cv=args.cv,
        scoring=args.scoring,
        random_state=args.random_state,
    )
    df_results["dataset"] = test_dataset_name
    if args.predefined_strategy is not None:
        df_results["strategy"] = args.predefined_strategy
    df_results["rule_strategy"] = args.rule_strategy
    df_results["enumeration_strategy"] = args.enumeration_strategy

    print("Dumping results to", args.output)
    df_results.to_pickle(args.output)

    print("Dumping enumerator statistics")
    with open(args.output + "-enumerator-statistics", "wb") as fout:
        enumerator.statistics.strategy = args.predefined_strategy
        pickle.dump(enumerator.statistics, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
