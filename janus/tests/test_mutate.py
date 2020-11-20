from collections import Counter

import tpot
import tpot.config

from janus.pipeline import pipeline_to_tree as pt
from janus.repair import mutate
from janus.repair.local_rules import (
    ComponentInsert,
    ComponentRemove,
    HyperparamRemove,
    ComponentUpdate,
    HyperparamUpdate,
    RULE_TYPES,
)

from tests.utils import (flatten, PipelineGenerator)

data = PipelineGenerator(nrows=100, ncols=10, seed=42)


def test_config_load():
    sampler = mutate.get_random_mutation_sampler()
    assert len(sampler.config) == len(tpot.config.classifier_config_dict)


def test_mutate_any_node():
    # should be able to mutate every node (except 'root')
    sampler = mutate.get_random_mutation_sampler()
    rule_counter = Counter()
    for p in data.pipelines:
        t = pt.to_tree(p)
        t.annotate()
        flat_nodes = flatten(t)

        for n in flat_nodes:
            rules = sampler.sample_rules(n, return_proba=True)
            # always has one rule per node
            assert len(rules) == 1
            rule, prob = rules[0]
            rule_counter[type(rule)] += 1
            if pt.is_root_node(n):
                assert rule is None
                assert prob == 0.0
                continue

            if pt.is_composition_op(n):
                # can only insert for these
                if n.label.endswith("Pipeline"):
                    assert isinstance(rule, ComponentInsert)
                if n.label.endswith("FeatureUnion"):
                    assert isinstance(rule, (ComponentInsert, ComponentRemove))

            if pt.is_param_node(n):
                assert n.parent is not None
                parent_label = n.parent.label
                param = n.payload[0]
                if parent_label not in sampler.config or param not in sampler.config[
                        parent_label]:
                    assert isinstance(rule, HyperparamRemove) or rule is None
                    if rule is None:
                        assert prob == 0.0
                    else:
                        assert prob > 0.0
                    continue

                if param == "score_func":
                    assert rule is None
                    assert prob == 0.0
                    continue

            if pt.is_component_node(n):
                if isinstance(rule, ComponentInsert):
                    assert pt.is_composition_node(n)

            # all others should have non-identity rule
            assert rule is not None
            assert prob > 0.0

    # should have all rule types (bad luck if don't....)
    for t in RULE_TYPES:
        assert rule_counter.get(t, 0.0) > 0


def test_node_prob():
    sampler = mutate.get_random_mutation_sampler()
    for p in data.pipelines:
        t = pt.to_tree(p)
        flat_nodes = flatten(t)
        all_probs = []
        for n in flat_nodes:
            prob = sampler.get_probability(rule=None, node=n)
            assert prob >= 0.0 and prob <= 1.0
            all_probs.append(prob)

    # would be terrible luck if this failed...
    assert max(all_probs) > 0.0
