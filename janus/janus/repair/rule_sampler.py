from collections import (
    Counter,
    defaultdict,
)

import numpy as np
import sklearn.base
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from janus.repair.local_rules import (
    HyperparamRule,
    ComponentRule,
    HyperparamUpdate,
    HyperparamRemove,
    ComponentUpdate,
    ComponentRemove,
    ComponentInsert,
    RuleCorpus,
)
from janus.repair.mutate import get_random_mutation_sampler
from janus.pipeline import pipeline_to_tree as pt

DEFAULT_SMALL_PROB = 1e-5


def get_node_key(n):
    if pt.is_component_node(n):
        return n.label
    elif pt.is_param_node(n):
        return n.payload[0]
    else:
        return None


def get_clf(obj):
    if isinstance(obj, Pipeline):
        return get_clf(obj.steps[-1][1])
    elif sklearn.base.is_classifier(obj):
        return obj
    else:
        raise Exception("Unhandled obj" + str(type(obj)))


class RuleSampler(object):
    def sample_rules(self, node, *args, **kwargs):
        raise NotImplementedError("Define in subclass")


class BaseUnweightedRuleSampler(RuleSampler):
    def __init__(self, rules, random_state=None):
        self.rule_map = defaultdict(lambda: [])
        if isinstance(rules, RuleCorpus):
            rules = rules.rules
        if random_state is None:
            self.random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState()
            random_state.set_state(np.random.get_state())
            self.random_state = random_state

        added = set()
        for r in rules:
            pre_node_key = get_node_key(r.pre)
            rule_key = r.key()
            if isinstance(r, HyperparamRule):
                if rule_key in added:
                    # for hyperparameters the rule_key is enough
                    # since there is no subtree change
                    continue
            added.add(rule_key)
            self.rule_map[pre_node_key].append(r)


class RandomRuleSampler(BaseUnweightedRuleSampler):
    def sample_rules(self, c, random_state=None):
        node_key = get_node_key(c)
        # can always applied identity
        candidate_rules = [None]
        if node_key is None or node_key not in self.rule_map:
            return candidate_rules

        candidate_rules.extend(self.rule_map[node_key])
        if random_state is not None:
            np.random.RandomState(random_state).shuffle(candidate_rules)
        else:
            self.random_state.shuffle(candidate_rules)
        return candidate_rules


class ScoreRuleSampler(BaseUnweightedRuleSampler):
    def sample_rules(self, c, random_state=None):
        node_key = get_node_key(c)
        # can always applied identity
        candidate_rules = [None]
        if node_key is None or node_key not in self.rule_map:
            return candidate_rules

        # sort rules in decreasing order of score delta
        mined_rules = sorted(
            self.rule_map[node_key],
            key=lambda x: x.score_delta(),
            reverse=True,
        )
        candidate_rules.extend(mined_rules)
        return candidate_rules


class WeightedRuleSampler(RuleSampler):
    def __init__(self, rules, random_state=None):
        rule_counters = defaultdict(lambda: Counter())
        key_to_rules = defaultdict(lambda: [])

        if isinstance(rules, RuleCorpus):
            rules = rules.rules
        for r in rules:
            node_key = get_node_key(r.pre)
            rule_key = r.key()
            rule_counters[node_key][rule_key] += 1
            key_to_rules[rule_key].append(r)

        self.rule_map = {}
        self.conditional_rule_probs = {}
        self.pre_probs = {}

        pre_normalizer = 0.0
        for pre_key, cter in rule_counters.items():
            rule_normalizer = 0.0
            weighted_rules = []
            for rule_key, ct in cter.items():
                rs = key_to_rules[rule_key]
                # take best by score delta
                rs = sorted(rs, key=lambda r: r.score_delta(), reverse=True)
                r = rs[0]
                weighted_rules.append((r, ct))
                rule_normalizer += ct
            weighted_rules = sorted(
                weighted_rules,
                key=lambda x: x[1],
                reverse=True,
            )
            weighted_rules = [(r, float(v) / rule_normalizer)
                              for r, v in weighted_rules]
            just_rules, just_weights = zip(*weighted_rules)
            self.rule_map[pre_key] = list(just_rules)
            self.conditional_rule_probs[pre_key] = dict(weighted_rules)
            self.pre_probs[pre_key] = rule_normalizer
            pre_normalizer += rule_normalizer

        self.pre_probs = {
            k: (float(v) / pre_normalizer)
            for k, v in self.pre_probs.items()
        }
        if random_state is None:
            self.random_state = np.random.RandomState(random_state)
        else:
            random_state = np.random.RandomState()
            random_state.set_state(np.random.get_state())
            self.random_state = random_state

    def sample_rules(self, c, return_proba=False, random_state=None):
        node_key = get_node_key(c)
        # can always applied identity
        candidate_rules = [None]
        if node_key is None or node_key not in self.rule_map:
            if return_proba:
                candidate_rules[0] = (candidate_rules[0], DEFAULT_SMALL_PROB)
            return candidate_rules

        candidate_rules.extend(self.rule_map[node_key])
        if return_proba:
            with_proba = []
            assert node_key in self.conditional_rule_probs
            prob_table = self.conditional_rule_probs[node_key]
            for r in candidate_rules:
                if r is None:
                    p = DEFAULT_SMALL_PROB
                else:
                    p = prob_table.get(r, 0.0)
                with_proba.append((r, p))
            return with_proba
        else:
            return candidate_rules

    def get_probability(self, rule=None, node=None):
        if rule is not None and node is not None:
            # conditional probability P(rule | node)
            k = get_node_key(node)
            if k not in self.conditional_rule_probs:
                return 0.0
            return self.conditional_rule_probs[k].get(rule, 0.0)
        elif node is not None and rule is None:
            return self.pre_probs.get(get_node_key(node), 0.0)
        else:
            raise ValueError("Undefined probability computation")


class RulePredictor(object):
    def __init__(self, feats_from_node, predictor):
        self.pipeline = None
        self.predictor = predictor
        self.feats_from_node = feats_from_node
        self.label_to_rule = {}
        self.pre_counts = defaultdict(lambda: 0.0)
        self.pre_probs = {}
        self.is_empty = True

    def fit(self, rules):
        X_data = []
        ys = []
        self.label_to_rule = {}

        pre_normalizer = 0.0
        for i, r in enumerate(rules):
            x = r.info()
            pre_key = get_node_key(r.pre)
            self.pre_counts[pre_key] += 1
            pre_normalizer += 1
            X_data.append(x)
            key_rule = r.key()
            y_label = str(key_rule)
            if y_label in self.label_to_rule:
                if r.score_delta() > self.label_to_rule[y_label].score_delta():
                    # better score, so keep this one for this key
                    self.label_to_rule[y_label] = r
            else:
                self.label_to_rule[y_label] = r
            ys.append(y_label)

        self.pre_probs = {
            k: float(v) / pre_normalizer
            for k, v in self.pre_counts.items()
        }

        vectorizer = DictVectorizer()
        self.pipeline = Pipeline([("featurize", vectorizer),
                                  ("pred", self.predictor)])
        if len(X_data) > 0:
            self.pipeline.fit(X_data, ys)
            self.is_empty = False
        return self

    def predict(self, node, return_proba=False):
        candidate_rules = [(None, DEFAULT_SMALL_PROB)]
        if not self.is_empty:
            feats = self.feats_from_node(node)
            predicted_probas = self.pipeline.predict_proba(feats)
            assert predicted_probas.shape[0] == 1, "One node at a time"
            predicted_probas = predicted_probas[0]
            clf = get_clf(self.pipeline)

            weighted_labels = sorted(
                zip(clf.classes_, predicted_probas),
                key=lambda x: x[1],
                reverse=True,
            )
            predicted_rules = [(self.label_to_rule[label], prob)
                               for label, prob in weighted_labels]
            candidate_rules.extend(predicted_rules)
        if return_proba:
            return candidate_rules
        else:
            return [r for r, _ in candidate_rules]

    def get_probability(self, rule=None, node=None):
        if rule is not None and node is not None:
            # conditional probability P(rule | node)
            rules_and_probs = self.predict(node, return_proba=True)
            for r, p in rules_and_probs:
                if r == rule:
                    return p
            return 0.0
        elif node is not None and rule is None:
            return self.pre_probs.get(get_node_key(node), 0.0)
        else:
            raise ValueError("Undefined probability computation")


class PredictiveRuleSampler(RuleSampler):
    def __init__(
        self,
        rules,
        component_predictor=None,
        hyperparam_predictor=None,
        random_state=None,
    ):
        if isinstance(rules, RuleCorpus):
            rules = rules.rules

        hyperparam_rules = []
        component_rules = []

        for r in rules:
            if isinstance(r, HyperparamRule):
                hyperparam_rules.append(r)
            elif isinstance(r, ComponentRule):
                component_rules.append(r)
            else:
                raise Exception("Unknown rule type" + str(type(r)))

        if hyperparam_predictor is None:
            hyperparam_predictor = RandomForestClassifier(
                random_state=random_state, )

        self.hyperparam_rule_predictor = RulePredictor(
            HyperparamRule.info_from_node,
            hyperparam_predictor,
        )
        self.hyperparam_rule_predictor.fit(hyperparam_rules)

        if component_predictor is None:
            component_predictor = RandomForestClassifier(
                random_state=random_state, )
        self.component_rule_predictor = RulePredictor(
            ComponentRule.info_from_node,
            component_predictor,
        )
        self.component_rule_predictor.fit(component_rules)
        pre_counts = dict(self.hyperparam_rule_predictor.pre_counts)
        pre_counts.update(self.component_rule_predictor.pre_counts)
        normalizer = float(sum(pre_counts.values()))
        self.pre_probs = {k: v / normalizer for k, v in pre_counts.items()}

    def sample_rules(self, n, return_proba=False, random_state=None):
        rules = [None]
        if pt.is_component_node(n):
            rules.extend(
                self.component_rule_predictor.predict(
                    n, return_proba=return_proba))
        if pt.is_param_node(n):
            rules.extend(
                self.hyperparam_rule_predictor.predict(
                    n, return_proba=return_proba))
        if return_proba:
            # set small prob to None
            rules[0] = (rules[0], DEFAULT_SMALL_PROB)
        return rules

    def get_probability(self, rule=None, node=None):
        assert rule is not None or node is not None
        if rule is None and node is not None:
            # prior probs should account for both types of nodes
            return self.pre_probs.get(get_node_key(node), 0.0)

        if pt.is_component_node(node):
            return self.component_rule_predictor.get_probability(rule, node)
        if pt.is_param_node(node):
            return self.hyperparam_rule_predictor.get_probability(rule, node)


def get_rule_sampler(strategy, corpus, random_state=None):
    if strategy == "random":
        rule_sampler = RandomRuleSampler(corpus, random_state=random_state)
    elif strategy == "score":
        rule_sampler = ScoreRuleSampler(corpus, random_state=random_state)
    elif strategy == "weighted":
        rule_sampler = WeightedRuleSampler(corpus, random_state=random_state)
    elif strategy == "predictive":
        rule_sampler = PredictiveRuleSampler(corpus, random_state=random_state)
    elif strategy == "mutation":
        rule_sampler = get_random_mutation_sampler()
    else:
        raise Exception("NYI sampler")
    return rule_sampler
