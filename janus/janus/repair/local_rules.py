#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import pickle
import sys

import numpy as np
import sklearn.base
from tpot.builtins import StackingEstimator
import tqdm
import zss

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.tree_pairs import CorpusEntry, TreePairCorpus


def is_update_edit(edit):
    return edit.type == zss.compare.Operation.update


def is_match_edit(edit):
    return edit.type == zss.compare.Operation.match


def is_remove_edit(edit):
    return edit.type == zss.compare.Operation.remove


def is_insert_edit(edit):
    return edit.type == zss.compare.Operation.insert


def get_safe_label(n):
    if n is not None:
        return n.label
    else:
        return None


def get_edit_op_str(op):
    type_strs = {
        zss.Operation.update: "Update",
        zss.Operation.remove: "Remove",
        zss.Operation.insert: "Insert",
    }
    return type_strs.get(op.type)


class LocalEditRule(object):
    @staticmethod
    def can_build_rule(self, edit):
        raise NotImplementedError()

    def can_apply(self, node):
        raise NotImplementedError()

    def apply(self, node):
        raise NotImplementedError()

    def key(self):
        raise NotImplementedError()

    def info(self):
        return self._info

    def score_delta(self):
        return self._score_delta

    def set_score_delta(self, delta):
        self._score_delta = delta

    def as_str(self):
        _, pre, ctx, post = self.key()
        cls = self.__class__.__name__
        if pre is not None:
            pre = pre.split(".")[-1]
        if post is not None:
            post = post.split(".")[-1]
        ctx = ", ".join(
            [e.split(".")[-1] if e is not None else "None" for e in ctx])

        s = "{cls}({pre} -> {post}) @ ({ctx})".format(
            cls=cls,
            pre=pre,
            post=post,
            ctx=ctx,
        )
        return s


def value_as_feature(val):
    if not isinstance(val, (str, int, float, bool)):
        return str(type(val))
    if isinstance(val, float) and np.isnan(val):
        return str(val)
    return val


class HyperparamRule(LocalEditRule):
    @staticmethod
    def info_from_node(node):
        info = {
            "parent": get_safe_label(node.parent),
            "hyperparam": node.payload[0],
            "hypervalue": value_as_feature(node.payload[1]),
            "hypervalue_type": str(type(node.payload[1])),
        }
        for c in node.siblings():
            try:
                if pt.is_param_node(c):
                    name, val = c.payload
                    info["sibling_param_name_" + name] = True
                    info["sibling_param_value_" + name] = value_as_feature(val)
                elif pt.is_component_node(c):
                    info["sibling_component_" + c.label] = True
                else:
                    pass
            except:
                pass
        return info


class ComponentRule(LocalEditRule):
    @staticmethod
    def info_from_node(node):
        info = {
            "parent": get_safe_label(node.parent),
            "component": node.label,
        }
        for c in node.siblings():
            if pt.is_component_node(c):
                info[c.label] = True
        return info


class HyperparamUpdate(HyperparamRule):
    def __init__(self, edit):
        self.pre, self.post = edit.arg1, edit.arg2
        self._info = HyperparamRule.info_from_node(self.pre)

    @staticmethod
    def can_build_rule(edit):
        return is_update_edit(edit) and \
            pt.is_param_node(edit.arg1) and \
            pt.is_param_node(edit.arg2) and \
            edit.arg1.parent is not None and \
            edit.arg2.parent is not None and \
            edit.arg1.parent.label == edit.arg2.parent.label

    def key(self):
        ctx = frozenset([self.pre.parent.label])
        return (str(type(self)), self.pre.payload[0], ctx, self.post.label)

    def can_apply(self, node):
        if not pt.is_param_node(
                node) or node.payload[0] != self.pre.payload[0]:
            return False

        if self.pre.parent.label != get_safe_label(node.parent):
            # hyperparameters depend on the parent component
            # so if mismatched, can't really apply
            return False

        # for string hypervalues (most likely representing enumerations)
        # we apply only if there is a match
        pre_hypervalue = self.pre.payload[1]
        post_hypervalue = self.post.payload[1]
        cand_hypervalue = node.payload[1]

        if post_hypervalue == cand_hypervalue:
            # no-op, don't bother applying
            return False

        if isinstance(pre_hypervalue, str):
            return pre_hypervalue == cand_hypervalue
        else:
            # TODO: for numeric we could add predicate or some form of abstraction
            # could be learned...
            return True

    def apply(self, node):
        return pt.shallow_copy(self.post)


class HyperparamRemove(HyperparamRule):
    # NB: removing a hyperparameter is equivalent to
    # setting it to its default value in the constructor
    def __init__(self, edit):
        self.pre, self.post = edit.arg1, edit.arg2
        self._info = HyperparamRule.info_from_node(self.pre)

    @staticmethod
    def can_build_rule(edit):
        return is_remove_edit(edit) and \
            pt.is_param_node(edit.arg1) and \
            edit.arg1.parent is not None and \
            edit.arg2 is None

    def key(self):
        ctx = frozenset([self.pre.parent.label])
        return (str(type(self)), self.pre.payload[0], ctx, None)

    def can_apply(self, node):
        if not pt.is_param_node(
                node) or node.payload[0] != self.pre.payload[0]:
            return False

        if self.pre.parent.label != get_safe_label(node.parent):
            # requires component context to effectively apply
            return False

        # for string hypervalues (most likely representing enumerations)
        # we apply only if there is a match
        pre_hypervalue = self.pre.payload[1]
        cand_hypervalue = node.payload[1]

        if isinstance(pre_hypervalue, str):
            return pre_hypervalue == cand_hypervalue
        else:
            # TODO: for numeric we could add predicate or some form of abstraction
            # could be learned...
            return True

    def apply(self, node):
        # we delete by returning None...
        # actual deletion is handled by caller
        return None


class ComponentUpdate(ComponentRule):
    def __init__(self, edit):
        self.pre, self.post = edit.arg1, edit.arg2

        compiled_post = pt.to_pipeline(self.post)
        self.post_is_classifier = sklearn.base.is_classifier(compiled_post)

        if self.post_is_classifier:
            wrapped_post = pt.to_tree(StackingEstimator(compiled_post))
            self.wrapped_post = wrapped_post.children[0]

        self._info = ComponentRule.info_from_node(self.pre)

    @staticmethod
    def can_build_rule(edit):
        base_cond = is_update_edit(edit) and \
            pt.is_component_node(edit.arg1) and \
            pt.is_component_node(edit.arg2) and \
            edit.arg1.parent is not None

        if not base_cond:
            return False
        # want to also try compiling the post on its own
        try:
            pt.to_pipeline(edit.arg2)
            return True
        except:
            return False

    def key(self):
        ctx = frozenset([
            self.pre.parent.label,
            get_safe_label(self.pre.left),
            get_safe_label(self.pre.right)
        ])
        return (str(type(self)), self.pre.label, ctx, self.post.label)

    def can_apply(self, node):
        if not pt.is_component_node(node) or node.label != self.pre.label:
            return False

        if node.label == self.post.label:
            # no-op
            return False

        # at least one direct neighbor matches
        return self.pre.parent.label == get_safe_label(node.parent) or \
            get_safe_label(self.pre.left) == get_safe_label(node.left) or \
            get_safe_label(self.pre.right) == get_safe_label(node.right)

    def apply(self, node):
        compiled_node = pt.to_pipeline(node)
        if self.post_is_classifier and not sklearn.base.is_classifier(
                compiled_node):
            post = self.wrapped_post
        else:
            post = self.post
        return pt.shallow_copy(post)


class ComponentRemove(ComponentRule):
    def __init__(self, edit):
        self.pre, self.post = edit.arg1, edit.arg2
        self._info = ComponentRule.info_from_node(self.pre)

    @staticmethod
    def can_build_rule(edit):
        return is_remove_edit(edit) and \
            pt.is_component_node(edit.arg1) and \
            edit.arg2 is None and \
            edit.arg1.parent is not None

    def key(self):
        ctx = frozenset([
            self.pre.parent.label,
            get_safe_label(self.pre.left),
            get_safe_label(self.pre.right)
        ])
        return (str(type(self)), self.pre.label, ctx, None)

    def can_apply(self, node):
        if not pt.is_component_node(node) or node.label != self.pre.label:
            return False

        return self.pre.parent.label == get_safe_label(node.parent) or \
            get_safe_label(self.pre.left) == get_safe_label(node.left) or \
            get_safe_label(self.pre.right) == get_safe_label(node.right)

    def apply(self, node):
        return None


class ComponentInsert(ComponentRule):
    # NB: for insertion, we insert
    # into the children of `node` we're applying rule to
    # and return the node with modified children
    def __init__(self, edit):
        # the "pre" is really the parent node
        self.pre = edit.pre_parent
        self.post = edit.arg2

        # some re-used info for pre children
        self.pre_children_n = len(self.pre.children)
        self.pre_children_labels = [c.label for c in self.pre.children]
        self.pre_children_label_set = set(self.pre_children_labels)

        compiled_post = pt.to_pipeline(self.post)
        self.post_is_classifier = sklearn.base.is_classifier(compiled_post)

        if self.post_is_classifier:
            wrapped_post = pt.to_tree(StackingEstimator(compiled_post))
            # remove the 'root' node, just want the one for the clf
            self.wrapped_post = wrapped_post.children[0]

        self._info = ComponentRule.info_from_node(self.pre)

    @staticmethod
    def can_build_rule(edit):
        base_cond = is_insert_edit(edit) and \
            edit.arg1 is None and \
            edit.pre_parent is not None and \
            pt.is_component_node(edit.arg2) and \
            edit.arg2.parent is not None

        if not base_cond:
            return False
        # want to also try compiling the post on its own
        try:
            pt.to_pipeline(edit.arg2)
            return True
        except:
            return False

    def key(self):
        ctx = frozenset([c.label for c in self.pre.children])
        return (str(type(self)), self.pre.label, ctx, self.post.label)

    def can_apply(self, node):
        # a component insertion can only be applied to
        # a "combinator" object, so must be Pipeline type
        # TODO: extend with other combinator types here if necessary
        if not pt.is_composition_node(node):
            return False

        # node must be a well-formed pipeline, i.e. there must be
        # a classifier at the end
        if len(node.children) == 0:
            return False

        try:
            compiled_possible_clf = pt.to_pipeline(node.children[-1])
            if not sklearn.base.is_classifier(compiled_possible_clf):
                return False
        except:
            # can't compile it, can't really insert appropriately
            return False

        # we apply it by inserting into its children
        # so check that at least one child matches
        # what we observed in the pre node's children
        return any(
            c.label in self.pre_children_label_set for c in node.children)

    def apply(self, node, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # find children of node that existed as a child of the pre
        candidate_locations = []
        for ix, c in enumerate(node.children):
            if c.label in self.pre_children_label_set:
                candidate_locations.append(ix)

        if len(candidate_locations) == 0:
            # only happens if we try to apply without
            # calling .can_apply, so must be trying to force
            # application...so we'll just set candidate_locations
            # to be any
            candidate_locations = np.arange(0, len(node.children))

        # pick a location at random
        target_ix = np.random.choice(candidate_locations, 1)[0]
        # randomly pick if insert before or after that ix
        # if target_ix == ix, we're insert before
        # so add 0 to insert before or 1 to insert after
        target_ix = target_ix + np.random.choice([0, 1], 1)[0]

        node = pt.shallow_copy(node)
        n_children = len(node.children)
        post = pt.shallow_copy(self.post)

        if target_ix < n_children:
            # the new component will *not* be at the
            # end of the pipeline
            # so if its a classifier
            # we want to insert, need to wrap in stackingestimator
            if self.post_is_classifier:
                post = pt.shallow_copy(self.wrapped_post)
        else:
            # at the end of the pipeline
            if not self.post_is_classifier:
                # can't have a non-classifier at the end of the
                # pipeline
                # so shift the insertion point back by one
                target_ix -= 1
            else:
                # the post is a classifier, so the existing
                # classifier needs to be wrapped in stacking estimator
                # otherwise pipeline is invalid
                existing_clf_node = node.children[-1]
                compiled_clf = pt.to_pipeline(existing_clf_node)
                has_classifier = sklearn.base.is_classifier(compiled_clf)
                # should always be true given the .can_apply condition
                assert has_classifier
                # we want to insert a new classifier at the end
                # so we take existing classifier and wrap it
                wrapped_clf = pt.to_tree(StackingEstimator(compiled_clf))
                wrapped_clf = wrapped_clf.children[0]
                # replace the raw clf with the new wrapped clf
                node.replace_child(n_children - 1, wrapped_clf)

        node.insert_child(target_ix, post)
        return node


RULE_TYPES = [
    HyperparamUpdate,
    HyperparamRemove,
    ComponentUpdate,
    ComponentRemove,
    ComponentInsert,
]


def edit_to_str(edit):
    pre_label = get_safe_label(edit.arg1)
    post_label = get_safe_label(edit.arg2)

    if pre_label is not None:
        pre_label = pre_label.split(".")[-1]

    if post_label is not None:
        post_label = post_label.split(".")[-1]

    return "{op}({pre} -> {post})".format(
        op=get_edit_op_str(edit),
        pre=pre_label,
        post=post_label,
    )


# add some more info to the edit obj
class AugmentedEdit(object):
    def __init__(self, edit, parent_match_edit=None):
        self._edit = edit

        if is_update_edit(edit) or is_remove_edit(edit):
            self.pre_parent = edit.arg1.parent

        if is_insert_edit(edit):
            assert parent_match_edit is not None
            assert is_match_edit(parent_match_edit)
            self.pre_parent = parent_match_edit.arg1

    def __getattr__(self, attr):
        return getattr(self._edit, attr)


def get_parent_match_edit(edit, all_edits):
    # need to find the parent where we inserted
    # the new node, so we find the "match"
    # associated with that... this means we only
    # consider insertions for nodes that match
    # which is a restriction of the search space
    # (but ok for our use case)
    possible_parent_match_edits = [
        e for e in all_edits if e.arg2 == edit.arg2.parent and is_match_edit(e)
    ]
    parent_match_edit = None
    if len(possible_parent_match_edits) == 1:
        parent_match_edit = possible_parent_match_edits[0]
    return parent_match_edit


class RuleCorpus(object):
    def __init__(self, pairs, max_edit_distance, exclude_params=None):
        self.rules = []
        self.max_edit_distance = max_edit_distance
        self.exclude_params = exclude_params
        self.build_rules(pairs)
        print("Extracted", len(self.rules), "local edit rules")
        rule_counter = Counter()
        rule_counter.update([type(r) for r in self.rules])
        print("Rule breakdown")
        for t, c in rule_counter.items():
            print("\ttype {}: {}".format(t.__name__, c))

    def build_rules(self, pairs):
        for pair in tqdm.tqdm(pairs):
            if pair.distance > self.max_edit_distance:
                continue
            for ix, edit in tqdm.tqdm(enumerate(pair.edits)):
                if is_match_edit(edit):
                    continue
                if is_insert_edit(edit):
                    parent_match_edit = get_parent_match_edit(edit, pair.edits)
                    if parent_match_edit is None:
                        # can't do anything if no parent...
                        continue
                else:
                    # not necessary info
                    parent_match_edit = None

                # edit with some more info
                aug_edit = AugmentedEdit(edit, parent_match_edit)

                if pair.pre.external_score is not None:
                    pair_score_delta = pair.post.external_score - pair.pre.external_score
                else:
                    # we do this by definition for cases where the pre fails
                    # TODO: consider if a different value makes more sense...
                    pair_score_delta = 0.0

                for rule_type in RULE_TYPES:
                    if rule_type.can_build_rule(aug_edit):
                        rule = rule_type(aug_edit)
                        rule.set_score_delta(pair_score_delta)
                        # some hyperparameters are not really worth modifying
                        if isinstance(rule, HyperparamRule
                                      ) and self.exclude_params is not None:
                            param_name = rule.pre.payload[0]
                            if param_name in self.exclude_params:
                                continue
                        self.rules.append(rule)


def get_args():
    parser = ArgumentParser(
        description="Extract local edit rules from tree pairs")
    parser.add_argument("--input", type=str, help="Path to pickled tree pairs")
    # no point in extracting local rules when in reality
    # the pre/post tree pairs require a ton of edits
    parser.add_argument(
        "--max_edit_distance",
        type=int,
        help="Max number of edits in the pair",
        default=3,
    )
    parser.add_argument(
        "--exclude_params",
        type=str,
        nargs="+",
        help="Hyperparams to exclude from update/remove rules",
        default=["random_state", "n_jobs", "verbose", "cv"])
    parser.add_argument("--output", type=str, help="Path to save local rules")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input, "rb") as fin:
        pairs = pickle.load(fin)

    rule_corpus = RuleCorpus(pairs, args.max_edit_distance,
                             args.exclude_params)

    with open(args.output, "wb") as fout:
        pickle.dump(rule_corpus, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
