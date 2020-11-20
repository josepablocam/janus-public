#!/usr/bin/env python3
import pytest

import numpy as np
import sklearn.datasets
import tpot
import tpot.config

from janus.pipeline import pipeline_to_tree as pt
from janus.repair import local_rules
from janus.repair.rule_sampler import get_node_key
from janus.search.random_search import RandomSearch


class PipelineGenerator(object):
    def __init__(self, nrows=100, ncols=10, seed=42, max_time_mins=1):
        self.pipelines = []

        print("Generating pipelines for tests")
        np.random.seed(seed)
        X, y = sklearn.datasets.make_classification(nrows, ncols)
        self.X = X
        self.y = y
        print("TPOT-generated pipelines")
        clf = tpot.TPOTClassifier(
            max_time_mins=max_time_mins,
            verbosity=3,
            random_state=seed,
        )
        clf.fit(self.X, self.y)
        ga_pipelines = [
            e["pipeline_obj"] for e in clf.evaluated_individuals_.values()
        ]
        assert len(ga_pipelines) > 0
        self.pipelines.extend(ga_pipelines)

        print("Random-search-generated pipeliens")
        clf2 = RandomSearch(
            tpot.config.classifier_config_dict,
            4,
            max_time_mins=max_time_mins,
            max_time_mins_per_pipeline=1,
            max_retries=100,
            cv=3,
            scoring="f1_macro",
            random_state=seed,
        )
        clf2.fit(self.X, self.y)
        rand_pipelines = [
            e["pipeline_obj"] for e in clf2.evaluated_individuals_.values()
        ]
        assert len(rand_pipelines) > 0
        self.pipelines.extend(rand_pipelines)
        np.random.shuffle(self.pipelines)


class PairedExample(object):
    def __init__(self, clf1, clf2):
        self.pipelines = [clf1, clf2]
        self.trees = [pt.to_tree(p) for p in self.pipelines]
        for t in self.trees:
            t.annotate()
        self.dist, self.edits = pt.tree_edit_distance(
            self.trees[0],
            self.trees[1],
            return_operations=True,
        )
        self.updates = []
        self.removes = []
        self.inserts = []
        for e in self.edits:
            if local_rules.is_update_edit(e):
                self.updates.append(e)
            elif local_rules.is_remove_edit(e):
                self.removes.append(e)
            elif local_rules.is_insert_edit(e):
                self.inserts.append(e)
            else:
                pass


def get_node_with_prefix(nodes, name):
    for n in nodes:
        if n.label.startswith(name):
            return n
    return None


def check_all_references_identical(o1, o2):
    if id(o1) != id(o2):
        return False
    children1 = o1.children
    children2 = o2.children
    if len(children1) != len(children2):
        return False
    for c1, c2 in zip(children1, children2):
        if not check_all_references_identical(c1, c2):
            return False
    return True


def flatten(tree, acc=None):
    if acc is None:
        acc = []
    acc.append(tree)
    for c in tree.children:
        acc.extend(flatten(c))
    return acc


class FakeRuleSampler(object):
    def __init__(self, rules, node_probs, cond_probs):
        self.rules = rules
        self.node_probs = node_probs
        self.cond_probs = cond_probs

    def sample_rules(self, node, return_proba=False):
        node_key = get_node_key(node)
        rules = self.rules.get(node_key, [None])
        if return_proba:
            rules = [(r, self.cond_probs.get(r, 0.0)) for r in rules]
        return rules

    def get_probability(self, node):
        return self.node_probs.get(get_node_key(node), 0.0)
