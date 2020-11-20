#!/usr/bin/env python3
import copy
import numpy as np
import tpot

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.local_rules import (
    HyperparamRemove,
    HyperparamUpdate,
    ComponentRemove,
    ComponentUpdate,
    ComponentInsert,
    get_safe_label,
)


class FakeEdit(object):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        self.pre_parent = None


class RandomMutationSampler(object):
    def __init__(self, config, random_state=None):
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState()
            random_state.set_state(np.random.get_state())
            self.random_state = random_state
        self.config = copy.deepcopy(config)
        expanded_config = {}

        for elem_name, elem_config in config.items():
            obj = self.sample_compiled_component(elem_name, tries=3)
            if obj is None:
                # failed to compile
                continue
            # need to expand names so we can resolve
            # when using our graph/compile etc
            expanded_elem_name = pt.to_tree(obj).children[0].label
            expanded_config[expanded_elem_name] = elem_config

        self.config = copy.deepcopy(expanded_config)
        print("Loaded {}/{} elements in config".format(len(self.config),
                                                       len(config)))

    def sample_value(self, vals):
        return self.random_state.choice(vals, 1)[0]

    def sample_full_config(self, elem_config):
        chosen = {}
        for k, possible_vals in elem_config.items():
            if isinstance(possible_vals, dict):
                ks = list(possible_vals.keys())
                if k == "score_func":
                    assert len(ks) == 1
                    assert ks[0] == "sklearn.feature_selection.f_classif"
                    assert possible_vals[ks[0]] is None
                    # nothing to do for this one, just leave it with
                    # default value
                    continue
                elif k == "estimator":
                    # recursively generate a configured object
                    # label
                    assert len(ks) == 1
                    assert ks[0].endswith("ExtraTreesClassifier")
                    component_config = possible_vals[ks[0]]
                    obj = self.sample_compiled_component(
                        ks[0],
                        component_config,
                        tries=3,
                    )
                    chosen[k] = obj
            else:
                chosen[k] = self.sample_value(possible_vals)
        return chosen

    def get_hyperparam_mutation(self, node):
        if node.payload[0] == "score_func":
            return None

        op = self.sample_value(["update", "remove"])
        if op == "remove":
            fake_edit = FakeEdit(node, None)
            return HyperparamRemove(fake_edit)
        else:
            assert node.parent is not None
            parent_config = self.config[node.parent.label]
            orig_possible_vals = parent_config[node.payload[0]]
            # remove self, want to force a change
            possible_vals = [
                v for v in orig_possible_vals if v != node.payload[1]
            ]
            if len(possible_vals) == 0:
                # if none, then we may as well return with this
                possible_vals = orig_possible_vals
            new_val = self.sample_value(possible_vals)
            new_node = pt.shallow_copy(node)
            new_node.payload = (node.payload[0], new_val)
            fake_edit = FakeEdit(node, new_node)
            return HyperparamUpdate(fake_edit)

    def sample_compiled_component(self, label, component_config=None, tries=3):
        func = pt.get_constructor(None, label=label)
        obj = None
        if component_config is None:
            component_config = self.config[label]
        while obj is None and tries > 0:
            try:
                tries -= 1
                func_config = self.sample_full_config(component_config)
                obj = func(**func_config)
            except Exception as err:
                pass
        return obj

    def get_component_mutation(self, node):
        if pt.is_composition_node(node):
            if node.label.endswith("Pipeline"):
                # don't want to remove self
                # any removals need to happen at the child level
                possible_ops = ["insert"]
            if node.label.endswith("FeatureUnion"):
                possible_ops = ["insert", "remove"]
        else:
            possible_ops = ["update", "remove"]
        op = self.sample_value(possible_ops)
        if op == "remove":
            fake_edit = FakeEdit(node, None)
            return ComponentRemove(fake_edit)
        else:
            possible_components = list(self.config.keys())
            if op == "insert":
                node = None

            if node is not None:
                # remove node.label from components, want to sample new
                possible_components = [
                    c for c in possible_components if c != node.label
                ]

            new_label = self.sample_value(possible_components)
            obj = self.sample_compiled_component(new_label, tries=3)
            if obj is None:
                raise Exception("Unable to sample new setting for " +
                                get_safe_label(node))
            obj_as_tree = pt.to_tree(obj)
            obj_as_tree.annotate()
            new_node = obj_as_tree.children[0]
            fake_edit = FakeEdit(node, new_node)

            if op == "update":
                return ComponentUpdate(fake_edit)
            else:
                # doesn't matter since we force apply
                # so just putting the root as the pre_parent
                fake_edit.pre_parent = obj_as_tree
                return ComponentInsert(fake_edit)

    def get_random_mutation(self, node):
        if pt.is_param_node(node):
            return self.get_hyperparam_mutation(node)
        elif pt.is_component_node(node) or pt.is_composition_node(node):
            return self.get_component_mutation(node)
        else:
            return None

    def sample_rules(self, n, return_proba=False, random_state=None):
        if pt.is_root_node(n):
            return [(None, 0.0)]
        else:
            try:
                random_mutation = self.get_random_mutation(n)
            except:
                random_mutation = None

            if random_mutation is None:
                # deprioritize identity ops
                random_prob = 0.0
            else:
                random_prob = self.random_state.uniform()

            return [(random_mutation, random_prob)]

    def get_probability(self, rule=None, node=None):
        if pt.is_root_node(node):
            return 0.0
        else:
            return self.random_state.uniform()


def get_random_mutation_sampler():
    return RandomMutationSampler(tpot.config.classifier_config_dict)
