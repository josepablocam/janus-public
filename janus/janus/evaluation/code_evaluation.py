#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
import tqdm

from janus.evaluation.synthetic_evaluation import run_single_tree
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
    get_tree_enumerator, )
from janus.repair.rule_sampler import (
    get_rule_sampler, )
from janus.repair.repairer import PipelineRepairer
from janus import utils
from janus import mp_utils


def parse_user_script(script_path):
    with open(script_path, "r") as fin:
        src = fin.read()
    g = dict(globals())
    l = {}
    exec (src, g, l)
    return l


class UserScript(object):
    def __init__(self, script_path):
        parsed = parse_user_script(script_path)
        self.path = script_path
        self.make_pipeline = parsed["make_pipeline"]
        self.dataset = parsed["dataset"]
        self.metric = parsed["metric"]


def get_dataset_seed(dataset_name):
    return ord(dataset_name[-1])


def run_evaluation(
        script_paths,
        enumerator,
        bound_num_repaired_pipelines,
        scoring=None,
        max_size=5000,
        dev_fraction=0.2,
        dev_cv=3,
        bound_k=3,
        cv=5,
        random_state=42,
):

    results = []
    for ix, script_path in tqdm.tqdm(enumerate(script_paths)):
        user_script = UserScript(script_path)

        X, y = utils.get_dataset(user_script.dataset)

        # same sampling/train/test split based on dataset
        dataset_seed = get_dataset_seed(user_script.dataset)
        utils.set_seed(dataset_seed)
        if X.shape[0] > max_size:
            sample_idx, _ = train_test_split(
                np.arange(0, X.shape[0]),
                train_size=max_size,
                random_state=dataset_seed,
                stratify=y,
            )
            X = X[sample_idx]
            y = y[sample_idx]

        X_rest, X_search, y_rest, y_search = train_test_split(
            X,
            y,
            test_size=dev_fraction,
            random_state=dataset_seed,
            stratify=y,
        )

        orig_pipeline = pt.to_tree(user_script.make_pipeline())
        result = run_single_tree(
            X_search,
            y_search,
            X_rest,
            y_rest,
            orig_pipeline,
            enumerator,
            bound_num_repaired_pipelines,
            dev_cv=dev_cv,
            bound_k=bound_k,
            cv=cv,
            scoring=user_script.metric if scoring is None else scoring,
            random_state=random_state + ix,
        )
        result["script_path"] = user_script.path
        result["dataset"] = user_script.dataset
        result["metric"] = user_script.metric
        result["id"] = ix
        # doesn't matter ... just adding for consistency
        # with what we output for synthetic_evaluation
        result["timestamp"] = np.random.random()
        results.append(result)

    df_results = pd.concat(results, axis=0)
    return df_results


def get_args():
    parser = ArgumentParser(
        description="Evaluate repairs on a set of script-based pipelines")
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
        "--scripts",
        type=str,
        nargs="+",
        help="Scripts with pipelines, dataset name, and metric",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1000,
        help="Max size of dataset (sample if above)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Alternative scoring metric",
    )
    parser.add_argument(
        "--bound_num_repaired_pipelines",
        type=int,
        help="Number of new (repaired) pipelines to eval",
        default=5,
    )
    parser.add_argument(
        "--bound_k",
        type=int,
        help="Enumerator-specific bound k (limiting number of rules applied)",
        default=3,
    )
    parser.add_argument(
        "--dev_fraction",
        type=float,
        help="Fraction of dataset to use as for dev/search",
        default=0.2,
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
        raise Exception("Unknown predefined_strategy: " +
                        args.predefined_strategy)

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
        force_apply=(args.rule_strategy == "mutation"))

    df_results = run_evaluation(
        args.scripts,
        enumerator,
        args.bound_num_repaired_pipelines,
        scoring=args.scoring,
        max_size=args.max_size,
        dev_fraction=args.dev_fraction,
        dev_cv=args.dev_cv,
        bound_k=args.bound_k,
        cv=args.cv,
        random_state=args.random_state,
    )

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
