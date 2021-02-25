from janus.repair.tree_enumerator import (
    get_tree_enumerator, )
from janus.repair.rule_sampler import (
    get_rule_sampler, )

import pickle

# named strategy -> (rule, enumeration)
PREDEFINED_STRATEGIES = {
    "weighted-transducer": ("weighted", "beam"), # janus
    "rf-transducer": ("predictive", "beam"), # deprecated
    "random-mutation": ("mutation", "orig-only"), # random-mutation baseline
    "random-janus": ("random", "orig-only"), # random-janus baseline
    "classifier-swap": (None, "classifier-swap"), # deprecated
    "meta-learning": ("mutation", "meta-learning"), # meta-learning baseline
    "meta-janus": ("weighted", "meta-janus"), # meta-janus approach
    "janus": ("weighted", "orig-only"),
}


def get_repair_tools(
        predefined_strategy=None,
        rule_strategy=None,
        enumeration_strategy=None,
        score_predictor=None,
        rules_paths=False,
        random_state=None,
):
    if predefined_strategy is not None:
        assert rule_strategy is None
        assert enumeration_strategy is None
        if predefined_strategy not in PREDEFINED_STRATEGIES:
            raise Exception("Unknown predefined_strategy: " +
                            predefined_strategy)
        rule_strategy, enumeration_strategy = PREDEFINED_STRATEGIES[
            predefined_strategy]

    rules = []
    if rule_strategy is not None and rule_strategy != "mutation":
        for p in rules_paths:
            with open(p, "rb") as fin:
                rule_corpus = pickle.load(fin)
                rules.extend(rule_corpus.rules)

    rule_sampler = get_rule_sampler(
        rule_strategy,
        rules,
        random_state,
    )
    if enumeration_strategy.startswith("meta") and score_predictor is not None:
        with open(score_predictor, "rb") as fin:
            score_predictor = pickle.load(fin)

    enumerator = get_tree_enumerator(
        enumeration_strategy,
        rule_sampler,
        force_apply=(rule_strategy == "mutation"),
        score_predictor=score_predictor,
    )

    return {
        "rules": rules,
        "rule_sampler": rule_sampler,
        "tree_enumerator": enumerator
    }
