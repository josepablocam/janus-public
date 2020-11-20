#!/usr/bin/env python3
import pytest

import numpy as np
import sklearn.linear_model
import sklearn.datasets
import tqdm

from janus.pipeline import pipeline_to_tree as pt

from tests.utils import PipelineGenerator

data = PipelineGenerator(nrows=100, ncols=10, seed=42)


def test_pipeline_to_tree():
    trees = [pt.to_tree(p) for p in tqdm.tqdm(data.pipelines)]
    assert not any(t is None for t in trees), "No empties allowed"
    assert all(t.label == "root" for t in trees), "All start with root"
    return trees


def test_tree_to_pipeline():
    orig_pipelines = data.pipelines
    trees = [pt.to_tree(p) for p in tqdm.tqdm(orig_pipelines)]
    regen_pipelines = [pt.to_pipeline(t) for t in tqdm.tqdm(trees)]
    regen_trees = [pt.to_tree(p) for p in regen_pipelines]

    for ix, (t1, t2) in enumerate(zip(trees, regen_trees)):
        j1 = pt.to_hashable_json(t1)
        j2 = pt.to_hashable_json(t2)
        assert j1 == j2, "Pipelines should match"


def test_tree_annotation():
    clf = sklearn.linear_model.LogisticRegression(penalty="l1")
    tr = pt.to_tree(clf)
    tr.annotate()
    assert tr.parent is None, "root has no parent"
    # check parents
    q = [tr]
    while len(q) > 0:
        last = q.pop()
        n = len(last.children)
        for ix, c in enumerate(last.children):
            assert c.parent == last, "parent/child relationship incorrect"
            left = None if ix == 0 else last.children[ix - 1]
            assert c.left == left, "left incorrect"

            right = None if (ix + 1) == n else last.children[ix + 1]
            assert c.right == right, "right incorrect"


def test_tree_replacement():
    clf = sklearn.linear_model.LogisticRegression(penalty="l1")
    tr = pt.to_tree(clf)

    clf2 = sklearn.linear_model.LinearRegression()
    tr2 = pt.to_tree(clf2)

    tr.replace_child(0, tr2.children[0])

    assert tr.children[0] == tr2.children[0], "replacement failed"
    assert pt.to_hashable_json(tr) == pt.to_hashable_json(
        tr2), "subtrees incorrect"


def test_tree_deletion():
    clf = sklearn.linear_model.LogisticRegression(penalty="l1")
    tr = pt.to_tree(clf)

    # delete all the hyperparameters --> equivalent to setting defaults
    # when we recompile into a pipeline
    hypers = list(tr.children[0].children)
    target_node = tr.children[0]
    n = len(hypers)
    for h in hypers:
        target_node.delete_child(h)
        with pytest.raises(ValueError):
            target_node.children.index(h)
        assert (n - 1) == len(target_node.children), "deletion failed"
        n -= 1
    assert n == 0, "deleting all children failed"

    check = pt.to_hashable_json(pt.to_tree(pt.to_pipeline(tr)))
    clf2 = sklearn.linear_model.LogisticRegression()
    answer = pt.to_hashable_json(pt.to_tree(clf2))
    assert check == answer, "deleting hyperparams didn't yield defaults"


def test_tree_insertion():
    clf = sklearn.linear_model.LogisticRegression(penalty="l1")
    tr = pt.to_tree(clf)

    test_label = "__fake_node_for_testing__"
    new_node = pt.mk_comp_node(test_label)

    # invalid indices
    with pytest.raises(AssertionError):
        tr.insert_child(len(tr.children) + 1, new_node)

    with pytest.raises(AssertionError):
        tr.insert_child(-1, new_node)

    tr.insert_child(0, new_node)

    assert tr.children[0].label == test_label
    assert tr.children[0].left is None
    assert tr.children[0].right == tr.children[1]
    assert tr.children[0].parent == tr
    assert tr.children[1].left == tr.children[0]
    assert tr.children[1].right is None
