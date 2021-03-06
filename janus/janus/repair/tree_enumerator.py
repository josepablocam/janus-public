#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import os
import itertools
import pickle
import sys
import heapq

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.ensemble
import sklearn.pipeline

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.local_rules import (  # noqa
    LocalEditRule, HyperparamUpdate, HyperparamRemove, ComponentUpdate,
    ComponentInsert, ComponentRemove, RuleCorpus, is_match_edit,
    get_safe_label, ComponentRule,
)
from janus.repair.rule_sampler import (
    get_rule_sampler,
    WeightedRuleSampler,
)
from janus.repair.meta_learning import PipelineScorePredictor
from janus import utils


def must_delete_subtree(tree, new_children):
    """
    Some subtrees become no-ops (or invalid) if they have
    no children, so check if that is the case
    """
    del_if_empty_combinators = [
        "sklearn.pipeline.Pipeline",
        "tpot.builtins.stacking_estimator.StackingEstimator"
    ]
    return (pt.is_root_node(tree) or
            (pt.is_component_node(tree)
             and tree.label in del_if_empty_combinators)) and all(
                 c is None for c in new_children)


def replace_children(tree, new_children, copy=True):
    if must_delete_subtree(tree, new_children):
        return None
    if copy:
        new_tree = pt.shallow_copy(tree)
    else:
        new_tree = tree
    return new_tree.set_children(new_children)


def replace_node(tree, orig_node, new_node):
    """
    Replace a node anywhere in the tree, and share references
    to unchanged portions of the tree
    """
    if tree == orig_node:

        if pt.is_root_node(tree):
            # we're at the root, so we should copy the entire tree
            new_tree = pt.deep_copy(tree)
            # root has a single child by pt.to_tree construction
            assert len(new_tree.children) == 1
            new_tree.replace_child(0, new_node)
            return new_tree, True

        return new_node, True
    else:
        any_change = False
        new_children = []
        for ix, c in enumerate(tree.children):
            new_c, changed = replace_node(
                c,
                orig_node,
                new_node,
            )
            new_children.append(new_c)
            if changed:
                # rest are unchanged
                new_children.extend(tree.children[(ix + 1):])
                any_change = True
                break
        if not any_change:
            # not on the path to our target replacement node
            # so can share all references
            return tree, False
        else:
            return replace_children(tree, new_children), True


class TreeEnumerator(object):
    def enumerate(self, tree, *args, **kwargs):
        raise NotImplementedError("Must implement in subclass")


class TopDownExhaustiveEnumerator(TreeEnumerator):
    """
    Exhaustively enumerates possible trees by traversing the tree top-down,
    applying rules to node, and recursively enumerating subtrees.
    """

    def __init__(self, rule_sampler):
        self.rule_sampler = rule_sampler

    def recursive_enumerate(self, node, rules_per_node=3):
        applied = 0
        rules = self.rule_sampler.sample_rules(node)
        for rule in rules:
            if applied >= rules_per_node:
                break
            if rule is None:
                new_node = node
            elif rule.can_apply(node):
                new_node = rule.apply(node)
                applied += 1
            else:
                continue

            if pt.is_param_node(new_node) or new_node is None:
                yield new_node
                continue

            # modify children of the transformed node
            # to make sure we have valid children
            subtree_gens = []
            for ix, child in enumerate(new_node.children):
                # generator for subtrees rooted at this child
                gen = self.recursive_enumerate(child, rules_per_node)
                subtree_gens.append(gen)

            # cartesian product over subtrees rooted at each child
            for subtrees in itertools.product(*subtree_gens):
                # insert this set of subtrees
                # and return to constitute a new tree
                # rooted at `new_node`
                new_tree = replace_children(
                    new_node,
                    subtrees,
                )
                yield new_tree

    def enumerate(self, pipeline, rules_per_node, X=None, y=None):
        tree = pt.to_tree(pipeline)
        tree.annotate()
        for candidate in self.recursive_enumerate(tree, rules_per_node):
            if candidate is None:
                # managed to delete all nodes (or critical nodes)
                continue
            try:
                compiled = pt.to_pipeline(candidate)
                if X is None:
                    yield compiled
                else:
                    try:
                        compiled.fit(X, y)
                        yield compiled
                    except:
                        continue
            except:
                continue


class EnumerationStatistics(object):
    def __init__(self):
        self.trace = []
        self.rule_type_counter = Counter()
        self.info = {
            "failures": [],
        }
        self.strategy = None

    def record(self, pipeline, lineage):
        self.trace.append((pipeline, frozenset(lineage)))
        for rule in lineage:
            self.rule_type_counter[type(rule)] += 1

    def record_failure(self, err):
        self.info["failures"].append(utils.get_exception_name(err))


class RepeatedBeamSearchEnumerator(TreeEnumerator):
    """
    Traverse tree and identify P(rule | node) * P(node) per rule,
    apply the top k rules
    Apply those edits. Append these tree to those to be returned.
    Then call again on these modified tree.
    Perform this up to a bound.
    Note that in contrast to standard beam search we don't prune,
    so the frontier grows exponentially.
    TODO: maybe we should actually prune...
    """

    def __init__(self, rule_sampler, force_apply):
        try:
            rule_sampler.get_probability
        except AttributeError:
            raise Exception(
                "Requires rule sampler that can compute probabilities")
        self.rule_sampler = rule_sampler
        self.force_apply = force_apply
        self.statistics = EnumerationStatistics()

    def collect_node_rule_probs(self, node, past_rules, acc=None):
        if acc is None:
            acc = {}
        rules_and_probs = self.rule_sampler.sample_rules(
            node, return_proba=True)
        # P(node)
        node_prior_prob = self.rule_sampler.get_probability(node=node)
        # only consider rules we can actually apply
        applicable_rules = []
        applicable_cond_probs = []
        for r, p in rules_and_probs:
            if r is None or self.force_apply or r.can_apply(node):
                applicable_rules.append(r)
                # P(rule | node)
                applicable_cond_probs.append(p)

        # renormalize cond_probs since not all rules may have been applicable
        if len(applicable_cond_probs) == 0 or np.max(
                applicable_cond_probs) == 0:
            acc[node] = (None, 0.0)
        else:
            applicable_cond_probs = np.array(applicable_cond_probs) / np.sum(
                applicable_cond_probs)

            best_prob = 0.0
            best_rule = None
            for r, p in zip(applicable_rules, applicable_cond_probs):
                if r in past_rules:
                    # don't pick a rule we have already used
                    # want diversity
                    continue
                if best_prob < p:
                    best_prob = p
                    best_rule = r

            # P(rule | node) * P(node)
            best_prob = best_prob * node_prior_prob
            acc[node] = (best_rule, best_prob)

        for c in node.children:
            self.collect_node_rule_probs(c, past_rules, acc)
        return acc

    def derive_variant_trees(self, tree, k, past_rules):
        tree.annotate()
        candidates = {}

        self.collect_node_rule_probs(tree, past_rules, acc=candidates)
        # in descending order of likelihood
        candidates = sorted(
            candidates.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )
        resulting_trees = []
        lineage = []
        k_ct = 0
        for ix, (node, (rule, prob)) in enumerate(candidates):
            if k_ct == k:
                break
            if prob == 0.0 or rule is None or rule in past_rules:
                # don't APPLY
                continue
            try:
                new_node = rule.apply(node)
            except AssertionError as err:
                if self.force_apply:
                    continue
                else:
                    raise err
            # replace new_node into tree
            new_tree, changed = replace_node(tree, node, new_node)

            assert changed
            resulting_trees.append(new_tree)
            new_tree_rules = set(past_rules)
            new_tree_rules.add(rule)
            lineage.append(new_tree_rules)
            k_ct += 1
        return resulting_trees, lineage

    def enumerate(self, pipeline, k):
        tree = pt.to_tree(pipeline)
        # add initial pipeline to avoid returning same unmodified
        already_handled = set()
        already_handled.add(pt.to_hashable_json(tree))

        tree.annotate()
        new_trees, lineage = self.derive_variant_trees(
            tree,
            k,
            past_rules=set(),
        )

        while len(new_trees) > 0:
            head_tree = new_trees[0]
            new_trees = new_trees[1:]

            head_lineage = lineage[0]
            lineage = lineage[1:]
            if head_tree is None:
                # bogus pipeline: had only one component
                # and we deleted it so it becomes None
                continue

            try:
                head_pipeline = pt.to_pipeline(head_tree)
            except Exception as err:
                # don't add to more new_trees
                # as has issue... (unlikely to fix
                # downstream)
                self.statistics.record_failure(err)
                continue

            # make sure compile to pipeline first to handle
            # things like default hyperparameter values
            h = pt.to_hashable_json(pt.to_tree(head_pipeline))
            if h in already_handled:
                continue
            already_handled.add(h)

            self.statistics.record(head_tree, head_lineage)
            yield head_tree

            rec_new_trees, rec_lineage = self.derive_variant_trees(
                head_tree,
                k,
                past_rules=set(head_lineage),
            )

            if len(rec_new_trees) > 0:
                new_trees.extend(rec_new_trees)
                lineage.extend(rec_lineage)
                # this tree is productive, so put it back
                # into the queue for later use (if possible)
                new_trees.append(head_tree)
                lineage.append(head_lineage)


class FixOriginalOnlyEnumerator(TreeEnumerator):
    """
    Don't perform beam, always attempt to repair only the original
    input pipeline
    """

    def __init__(self, rule_sampler, force_apply):
        try:
            rule_sampler.get_probability
        except AttributeError:
            raise Exception(
                "Requires rule sampler that can compute probabilities")
        self.rule_sampler = rule_sampler
        self.force_apply = force_apply
        self.statistics = EnumerationStatistics()

    def collect_node_rule_probs(self, node, past_rules, acc=None):
        if acc is None:
            acc = {}
        rules_and_probs = self.rule_sampler.sample_rules(
            node, return_proba=True)
        # P(node)
        node_prior_prob = self.rule_sampler.get_probability(node=node)
        # only consider rules we can actually apply
        applicable_rules = []
        applicable_cond_probs = []
        for r, p in rules_and_probs:
            if r is None or self.force_apply or r.can_apply(node):
                applicable_rules.append(r)
                # P(rule | node)
                applicable_cond_probs.append(p)

        # renormalize cond_probs since not all rules may have been applicable
        if len(applicable_cond_probs) == 0 or np.max(
                applicable_cond_probs) == 0:
            acc[node] = (None, 0.0)
        else:
            applicable_cond_probs = np.array(applicable_cond_probs) / np.sum(
                applicable_cond_probs)

            best_prob = 0.0
            best_rule = None
            for r, p in zip(applicable_rules, applicable_cond_probs):
                if r in past_rules:
                    # don't pick a rule we have already used
                    # want diversity
                    continue
                if best_prob < p:
                    best_prob = p
                    best_rule = r

            # P(rule | node) * P(node)
            best_prob = best_prob * node_prior_prob
            acc[node] = (best_rule, best_prob)

        for c in node.children:
            self.collect_node_rule_probs(c, past_rules, acc)
        return acc

    def derive_variant_trees(self, tree, past_rules):
        tree.annotate()
        candidates = {}

        self.collect_node_rule_probs(tree, past_rules, acc=candidates)
        # in descending order of likelihood
        candidates = sorted(
            candidates.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )
        resulting_trees = []
        lineage = []

        for ix, (node, (rule, prob)) in enumerate(candidates):
            if rule is None or rule in past_rules:
                # don't APPLY
                continue
            try:
                new_node = rule.apply(node)
            except AssertionError as err:
                if self.force_apply:
                    continue
                else:
                    raise err
            # replace new_node into tree
            new_tree, changed = replace_node(tree, node, new_node)

            assert changed
            resulting_trees.append(new_tree)
            new_tree_rules = set(past_rules)
            new_tree_rules.add(rule)
            lineage.append(new_tree_rules)
        return resulting_trees, lineage

    def enumerate(self, pipeline, k=None):
        orig_tree = pt.to_tree(pipeline)
        # add initial pipeline to avoid returning same unmodified
        already_handled = set()
        already_handled.add(pt.to_hashable_json(orig_tree))

        orig_tree.annotate()
        new_trees = []
        past_rules = set()

        while True:
            if len(new_trees) == 0:
                new_trees, lineage = self.derive_variant_trees(
                    orig_tree,
                    past_rules=past_rules,
                )
            else:
                head_tree = new_trees.pop(0)
                head_lineage = lineage.pop(0)

                past_rules.update(head_lineage)

                if head_tree is None:
                    # bogus pipeline: had only one component
                    # and we deleted it so it becomes None
                    continue

                try:
                    head_pipeline = pt.to_pipeline(head_tree)
                except Exception as err:
                    # don't add to more new_trees
                    # as has issue... (unlikely to fix
                    # downstream)
                    self.statistics.record_failure(err)
                    continue

                # make sure compile to pipeline first to handle
                # things like default hyperparameter values
                h = pt.to_hashable_json(pt.to_tree(head_pipeline))
                if h in already_handled:
                    continue
                already_handled.add(h)

                self.statistics.record(head_tree, head_lineage)
                yield head_tree


class FixedClassifierSwap(TreeEnumerator):
    """
    Replaces the classifier at the end of a pipeline with a
    pre-defined classifier object
    """

    def __init__(self, classifier=None):
        if classifier is None:
            classifier = sklearn.ensemble.RandomForestClassifier()
        assert sklearn.base.is_classifier(classifier), "must be classifier"
        self.classifier = classifier
        self.statistics = EnumerationStatistics()

    def enumerate(self, tree, *args, **kwargs):
        # yields a single tree
        # where we have replaced the final step in the pipeline
        # with self.classifier
        pipeline = pt.to_pipeline(tree)
        new_pipeline = sklearn.base.clone(pipeline)
        steps = new_pipeline.steps
        step_name, old_clf = steps[-1]
        if not sklearn.base.is_classifier(old_clf):
            # can't apply this if not ending in classifier
            self.statistics.record_failure(ValueError("last step not clf"))
            yield None
            return

        # replace with our fixed classifier
        steps[-1] = (step_name, sklearn.base.clone(self.classifier))
        new_pipeline.steps = steps
        new_tree = pt.to_tree(new_pipeline)

        orig_hash = pt.to_hashable_json(pt.to_tree(pipeline))
        if orig_hash == pt.to_hashable_json(new_tree):
            # by definition we can't produce just the same tree
            # back so return None
            self.statistics.record_failure(ValueError("last step unchanged"))
            yield None
            return

        # don't care about lineage -> fixed op, so ignore
        self.statistics.record(new_tree, [])
        yield new_tree


class MetaLearningTreeEnumerator(TreeEnumerator):
    """
    Enumerates trees using an base enumerator, scores the set of trees
    using a score predictor model and returns the tree with the highest
    predicted score
    """

    def __init__(self, score_predictor, base_enumerator):
        self.score_predictor = score_predictor
        self.base_enumerator = base_enumerator
        self.buffer = []
        # just reference the base enumerator's statistics
        self.statistics = base_enumerator.statistics

    def enumerate(self, tree, k, *args, **kwargs):
        # for each `next` called on this enumerator
        # we call the underlying tree enumerator for up to k
        # times, and maintain priority queue
        # where we return the best score so far
        self.buffer = []
        base_gen = self.base_enumerator.enumerate(tree, k)
        base_enum_done = False

        i = 0
        while True:
            # get up to K trees from the base enumerator
            # score them and yield largest
            ct = 0
            while not base_enum_done and ct < k:
                try:
                    candidate_tree = next(base_gen)
                except StopIteration:
                    base_enum_done = True
                    break
                ct += 1
                pred_score = self.score_predictor.predict([candidate_tree])[0]
                # min heap so push on negative of predicted score
                # use i to tie break rather than comparison on trees
                # which is not defined
                heapq.heappush(self.buffer, ((-pred_score, i), candidate_tree))
                i += 1

            # return best available if still have
            if len(self.buffer) > 0:
                entry = heapq.heappop(self.buffer)
                yield entry[1]
            else:
                return


def get_tree_enumerator(
        strategy,
        rule_sampler,
        force_apply=False,
        score_predictor=None,
):
    if strategy == "exhaustive":
        return TopDownExhaustiveEnumerator(rule_sampler)
    elif strategy == "beam":
        return RepeatedBeamSearchEnumerator(
            rule_sampler, force_apply=force_apply)
    elif strategy == "classifier-swap":
        return FixedClassifierSwap()
    elif strategy == "orig-only":
        return FixOriginalOnlyEnumerator(rule_sampler, force_apply=force_apply)
    elif strategy == "meta-learning":
        base_enumerator = get_tree_enumerator(
            "orig-only", rule_sampler, force_apply=force_apply)
        meta_enumerator = MetaLearningTreeEnumerator(
            score_predictor=score_predictor,
            base_enumerator=base_enumerator,
        )
        return meta_enumerator
    elif strategy == "meta-janus":
        base_enumerator = get_tree_enumerator(
            "beam", rule_sampler, force_apply=force_apply)
        meta_enumerator = MetaLearningTreeEnumerator(
            score_predictor=score_predictor,
            base_enumerator=base_enumerator,
        )
        return meta_enumerator
    else:
        raise ValueError("Unknown tree enumeration strategy:" + strategy)


def get_args():
    parser = ArgumentParser(description="Test apply....")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to pickled rule corpus",
    )
    parser.add_argument(
        "--enumeration_strategy",
        type=str,
        help="Rule application strategy (i.e. tree enumeration strategy)",
        default="exhaustive",
    )
    parser.add_argument(
        "--rule_strategy",
        type=str,
        help="Rule sampling strategy",
        default="random",
    )
    parser.add_argument(
        "--bound_num_pipelines",
        type=int,
        help="Max number of pipelines to return",
        default=10,
    )
    parser.add_argument(
        "--bound_k",
        type=int,
        help="Max number of rules to apply to a node",
        default=3,
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
    with open(args.input, "rb") as fin:
        corpus = pickle.load(fin)

    pipeline = sklearn.pipeline.Pipeline(
        [("clf", sklearn.linear_model.LogisticRegression())])

    rule_sampler = get_rule_sampler(args.rule_strategy, corpus,
                                    args.random_state)
    enumerator = get_tree_enumerator(args.enumeration_strategy, rule_sampler)
    orig_tree = pt.to_tree(pipeline)
    explored = set([orig_tree])

    ix = 0
    for p in enumerator.enumerate(pipeline, args.bound_k):
        if ix >= args.bound_num_pipelines:
            break
        new_tree = pt.to_tree(p)
        h = pt.to_hashable_json(new_tree)
        if h in explored:
            continue
        explored.add(h)

        print("New pipeline", ix)
        dist, edits = pt.tree_edit_distance(
            orig_tree,
            new_tree,
            return_operations=True,
        )
        print("Distance", dist)
        ct_edits = 0
        for edit in edits:
            if is_match_edit(edit):
                continue
            msg = "Edit: {} -> {}".format(
                get_safe_label(edit.arg1), get_safe_label(edit.arg2))
            print(msg)
            ct_edits += 1
        print(pt.to_json(new_tree))
        ix += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
