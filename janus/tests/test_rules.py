#!/usr/bin/env python3

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import sklearn.datasets

from janus.pipeline import pipeline_to_tree as pt
from janus.repair import local_rules

from tests.utils import (PairedExample, get_node_with_prefix)


def test_build_updates():
    ex = PairedExample(LogisticRegression(penalty="l2"),
                       LogisticRegression(penalty="l1"))
    assert ex.dist == 1, "Single difference failed"

    # test hyper parameter update
    hyper_update = ex.updates[0]
    assert local_rules.HyperparamUpdate.can_build_rule(hyper_update)

    ex2 = PairedExample(
        LogisticRegression(penalty="l2"),
        sklearn.linear_model.LinearRegression(),
    )
    assert ex2.dist > 0
    comp_update = next(e for e in ex2.updates
                       if e.arg1.label.endswith("Regression"))
    assert local_rules.ComponentUpdate.can_build_rule(comp_update)

    assert not local_rules.ComponentUpdate.can_build_rule(hyper_update)
    assert not local_rules.HyperparamUpdate.can_build_rule(comp_update)


def test_apply_updates():
    ex1 = PairedExample(LogisticRegression(penalty="l2"),
                        LogisticRegression(penalty="l1"))
    rule = local_rules.HyperparamUpdate(ex1.updates[0])
    p1 = get_node_with_prefix(ex1.trees[0].children[0].children,
                              "param_penalty")
    p2 = get_node_with_prefix(ex1.trees[1].children[0].children,
                              "param_penalty")
    assert p1 is not None and p2 is not None
    assert rule.apply(p1).label == p2.label

    ex2 = PairedExample(
        LogisticRegression(penalty="l2"),
        sklearn.linear_model.LinearRegression(),
    )
    comp_update = next(e for e in ex2.updates
                       if e.arg1.label.endswith("Regression"))
    rule = local_rules.ComponentUpdate(comp_update)
    p1 = get_node_with_prefix(ex1.trees[0].children, "sklearn.linear")
    p2 = get_node_with_prefix(ex2.trees[1].children, "sklearn.linear")
    assert p1 is not None and p2 is not None
    assert rule.apply(p1).label == p2.label


def test_removes():
    ex1 = PairedExample(
        Pipeline([("s0", MinMaxScaler()),
                  ("clf", LogisticRegression(penalty="l2"))]),
        Pipeline([("clf", LogisticRegression(penalty="l2"))]),
    )
    assert local_rules.ComponentRemove.can_build_rule(ex1.removes[0])

    p = LogisticRegression(penalty="l1")
    ptree = pt.to_tree(p)
    ptree.annotate()

    penalty_node = get_node_with_prefix(ptree.children[0].children,
                                        "param_penalty")
    assert penalty_node is not None
    nparams = len(ptree.children[0].children)
    ptree.children[0].delete_child(penalty_node)
    post_nparams = len(ptree.children[0].children)
    assert (nparams - post_nparams) == 1

    orig_ptree = pt.to_tree(p)
    orig_ptree.annotate()

    dist, edits = pt.tree_edit_distance(
        orig_ptree,
        ptree,
        return_operations=True,
    )
    assert dist == 1
    remove_edits = [e for e in edits if local_rules.is_remove_edit(e)]
    assert len(remove_edits) == 1
    remove_edit = remove_edits[0]

    assert local_rules.HyperparamRemove.can_build_rule(remove_edit)

    assert not local_rules.ComponentRemove.can_build_rule(remove_edit)
    assert not local_rules.HyperparamRemove.can_build_rule(ex1.removes[0])

    # apply them
    ptree = ex1.trees[0]
    comp_rule = local_rules.ComponentRemove(ex1.removes[0])
    comp_node = ptree.children[0].children[0]

    hyper_rule = local_rules.HyperparamRemove(remove_edit)
    hyper_node = penalty_node

    assert comp_rule.can_apply(comp_node)
    assert comp_rule.apply(comp_node) is None

    assert hyper_rule.can_apply(hyper_node)
    assert hyper_rule.apply(hyper_node) is None

    assert not comp_rule.can_apply(hyper_node)
    assert not hyper_rule.can_apply(comp_node)


def test_insert():
    ex1 = PairedExample(
        Pipeline([("clf", LogisticRegression(penalty="l2"))]),
        Pipeline([("s0", MinMaxScaler()),
                  ("clf", LogisticRegression(penalty="l2"))]),
    )
    parent_match_edit = local_rules.get_parent_match_edit(
        ex1.inserts[0],
        ex1.edits,
    )
    assert parent_match_edit is not None
    augedit = local_rules.AugmentedEdit(ex1.inserts[0], parent_match_edit)
    assert local_rules.ComponentInsert.can_build_rule(augedit)

    tree = ex1.trees[0]
    tree2 = ex1.trees[1]

    rule = local_rules.ComponentInsert(augedit)
    # the pipeline node
    assert rule.can_apply(tree.children[0])
    # a non pipeline node
    assert not rule.can_apply(tree.children[0].children[0])

    new_subtree = rule.apply(tree.children[0])
    assert len(new_subtree.children) == 2

    expected_label = ex1.trees[1].children[0].children[0].label
    assert new_subtree.children[0].label == expected_label
