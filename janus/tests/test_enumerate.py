#!/usr/bin/env python3

import copy
import sys
sys.path.append(".")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.pipeline import Pipeline

import janus.pipeline.pipeline_to_tree as pt
from janus.repair import tree_enumerator
from janus.repair.rule_sampler import get_node_key
from janus.repair.local_rules import (
    is_update_edit,
    HyperparamUpdate,
    is_remove_edit,
    ComponentRemove,
    is_insert_edit,
    get_parent_match_edit,
    AugmentedEdit,
    ComponentInsert,
)

from tests.utils import (
    FakeRuleSampler,
    get_node_with_prefix,
    check_all_references_identical,
    flatten,
)


def test_must_delete_subtree():
    p = Pipeline([("clf", LogisticRegression())])
    t = pt.to_tree(p)
    t.annotate()

    assert t.children[0].label.endswith("Pipeline")
    assert tree_enumerator.must_delete_subtree(t.children[0], [])
    assert tree_enumerator.must_delete_subtree(t.children[0], [None, None])
    assert not tree_enumerator.must_delete_subtree(
        t.children[0],
        t.children[0].children,
    )


def test_replace_children():
    p = Pipeline([("clf", LogisticRegression())])
    t = pt.to_tree(p)
    t.annotate()

    t2 = copy.deepcopy(t)
    assert tree_enumerator.replace_children(t2.children[0], []) is None

    t3 = copy.deepcopy(t)
    fake_nodes = [pt.mk_comp_node("fake1"), pt.mk_comp_node("fake2")]
    subtree = t3.children[0].children[0]
    assert subtree.label.endswith("Regression")

    new_subtree = tree_enumerator.replace_children(subtree,
                                                   fake_nodes,
                                                   copy=True)
    assert id(new_subtree) != id(subtree)
    assert len(new_subtree.children) == 2
    assert len(subtree.children) != 2
    assert new_subtree.children[0] == fake_nodes[0]
    assert new_subtree.children[1] == fake_nodes[1]

    new_subtree = tree_enumerator.replace_children(subtree,
                                                   fake_nodes,
                                                   copy=False)
    assert id(new_subtree) == id(subtree)
    assert len(new_subtree.children) == 2
    assert new_subtree.children[0] == fake_nodes[0]
    assert new_subtree.children[1] == fake_nodes[1]


def test_replace_node():
    p1 = Pipeline([("s0", MinMaxScaler()), ("clf", LogisticRegression())])
    p2 = Pipeline([("clf", LogisticRegression(penalty="l1"))])

    t1 = pt.to_tree(p1).annotate()
    t2 = pt.to_tree(p2).annotate()

    penalty1 = get_node_with_prefix(t1.children[0].children[-1].children,
                                    "param_penalty")
    penalty2 = get_node_with_prefix(t2.children[0].children[-1].children,
                                    "param_penalty")

    t1_modified, changed = tree_enumerator.replace_node(t1, penalty1, penalty2)
    assert changed
    # root is modified, so copy
    assert id(t1_modified) != id(t1)
    # pipeline node is modified so copy
    assert id(t1_modified.children[0]) != id(t1.children[0])
    # the s0 node is shared
    assert id(t1_modified.children[0].children[0]) == id(
        t1.children[0].children[0])
    # the clf node is modified so copy
    assert id(t1_modified.children[0].children[-1]) != id(
        t1.children[0].children[-1])
    # the node we inserted should be the right one
    new_node = get_node_with_prefix(
        t1_modified.children[0].children[-1].children, "param_penalty")
    assert new_node.label == "param_penalty:l1"
    assert new_node.payload == ("penalty", "l1")
    assert new_node.parent == t1_modified.children[0].children[-1]

    # if we try to insert a made up node (i.e. no match ever)
    # should produce no change in the tree
    fake_node = pt.mk_comp_node("fake")
    t2_modified, changed = tree_enumerator.replace_node(
        t2, fake_node, penalty1)
    assert not changed
    assert check_all_references_identical(t2, t2_modified)


def test_beam_enumerator():
    p1 = Pipeline([("clf", LogisticRegression(penalty="l2"))])
    p2 = Pipeline([("clf", LogisticRegression(penalty="l1"))])

    _, ops = pt.tree_edit_distance(p1, p2, return_operations=True)
    update_op = [o for o in ops if is_update_edit(o)][0]
    # rule 1: penalty=l2 -> penalty=l1
    r1 = HyperparamUpdate(update_op)

    p5 = Pipeline([("clf", LogisticRegression(penalty="elasticnet"))])
    _, ops = pt.tree_edit_distance(p1, p5, return_operations=True)
    update_op = [o for o in ops if is_update_edit(o)][0]
    # rule 1.5: penalty=l2 -> penalty=elasticnet
    r1_5 = HyperparamUpdate(update_op)

    p3 = Pipeline([("s0", MinMaxScaler()),
                   ("clf", LogisticRegression(penalty="l2"))])
    _, ops = pt.tree_edit_distance(p3, p1, return_operations=True)
    remove_op = [o for o in ops if is_remove_edit(o)][0]

    # rule 2: remove MinMaxScaler
    r2 = ComponentRemove(remove_op)

    p4 = Pipeline([("s0", StandardScaler()),
                   ("clf", LogisticRegression(penalty="l2"))])

    _, ops = pt.tree_edit_distance(p1, p4, return_operations=True)
    insert_op = [o for o in ops if is_insert_edit(o)][0]
    augedit = AugmentedEdit(insert_op, get_parent_match_edit(insert_op, ops))
    # rule 3: insert StandardScaler
    r3 = ComponentInsert(augedit)

    n1 = r1.pre
    n2 = r2.pre
    n3 = r3.pre
    rules = {
        get_node_key(n1): [r1, r1_5],
        get_node_key(n2): [r2],
        get_node_key(n3): [r3],
    }
    node_probs = {
        get_node_key(n1): 0.5,
        get_node_key(n2): 0.2,
        get_node_key(n3): 0.3,
    }
    cond_probs = {
        r1: 0.7,
        r1_5: 0.15,
        r2: 0.3,
        r3: 0.1,
    }

    rule_sampler = FakeRuleSampler(rules, node_probs, cond_probs)

    rs = rule_sampler.sample_rules(n1, return_proba=True)
    assert len(rs) == 2

    enumerator = tree_enumerator.RepeatedBeamSearchEnumerator(
        rule_sampler,
        force_apply=False,
    )

    # # should sort max to min prob by rules
    t1 = pt.to_tree(p1).annotate()
    opt_rules = enumerator.collect_node_rule_probs(t1, past_rules=[], acc={})
    flat_nodes = flatten(t1)
    assert len(opt_rules) == len(flat_nodes)

    # rule 1 is best for that node
    target_n = next(n for n in flat_nodes if n.label == r1.pre.label)
    opt_rule_and_prob = opt_rules[target_n]
    assert opt_rule_and_prob[0] == r1
    # we normalize the conditional probabilities to those that
    # can be applied:
    norm_cond_prob = cond_probs[r1] / (cond_probs[r1] + cond_probs[r1_5])
    expected_prob = node_probs[get_node_key(target_n)] * norm_cond_prob
    assert (opt_rule_and_prob[1] - expected_prob) < 1e-5

    # if we collect optimal node/rules again after using r1, we should get r1_5
    # for that node
    opt_rules = enumerator.collect_node_rule_probs(t1, past_rules=[r1], acc={})
    opt_rule_and_prob = opt_rules[target_n]
    assert opt_rule_and_prob[0] == r1_5
    # we normalize the conditional probabilities to those that
    # can be applied:
    norm_cond_prob = cond_probs[r1_5] / (cond_probs[r1] + cond_probs[r1_5])
    expected_prob = node_probs[get_node_key(target_n)] * norm_cond_prob
    assert (opt_rule_and_prob[1] - expected_prob) < 1e-5

    new_trees, lineage = list(
        enumerator.derive_variant_trees(t1, k=5, past_rules=[]))
    # at most 2 (even though k = 5)
    # penalty=l2->l1, insert(StandardScaler)
    assert len(new_trees) == 2
    assert list(lineage[0])[0] == r1
    assert list(lineage[1])[0] == r3

    gen = enumerator.enumerate(p1, k=5)
    trees_t1 = list(gen)
    # l2->l1, insert(StandardScaler), l2->elastic
    assert len(trees_t1) == 3

    gen = enumerator.enumerate(p5, k=5)
    trees_t2 = list(gen)
    # insert(StandardScaler)
    assert len(trees_t2) == 1

    gen = enumerator.enumerate(p3, k=10)
    trees_t3 = list(gen)
    # (Overall possible rules): outcome pipeline
    # l2 -> l1:                            MinMax, LR(l1)           (yes)
    # l2 -> elastic:                       MinMax, LR(elastic)
    # insert(StandardScaler):              MinMax, SS, LR(l2)       (yes)
    # remove(MinMaxScaler):                LR(L2)                   (yes)
    # insert(SS), l2->l1:                  MinMax, SS, LR(l1)       (yes)
    # insert(SS), l2->elastic:             MinMax, SS, LR(elastic)
    # remove(MM), l2->l1:                  LR(l1)                   (yes)
    # remove(MM), l2->elastic:             LR(elastic)
    # remove(MM), insert(SS)               SS, LR(l2)               (yes)
    # remove(MM), insert(SS), l2->l1:      SS, LR(l1)               (yes)
    # remove(MM), insert(SS), l2->elastic: SS, LR(elastic)
    assert len(trees_t3) < 11
