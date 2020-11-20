#!/usr/bin/env python3
import pytest

import numpy as np
import sklearn.linear_model
import sklearn.datasets
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.repairer import PipelineRepairer
from janus.repair.rule_sampler import get_rule_sampler
from janus.repair.tree_enumerator import get_tree_enumerator
from janus import utils

from tests.utils import PipelineGenerator
import tqdm

import copy
import sys
sys.path.append(".")

data = PipelineGenerator(nrows=100, ncols=10, seed=42, max_time_mins=0.5)


def get_repair_hashes(repairer, seed, num_passes=2, num_pipelines=3):
    passes = [[]] * num_passes
    for i in range(0, num_passes):
        print("Pass: {}".format(i))
        utils.set_seed(seed)
        num_remaining = num_pipelines
        pbar = tqdm.tqdm(total=num_pipelines)
        for p in data.pipelines:
            if num_remaining <= 0:
                break
            repaired = repairer.repair(p, data.X, data.y, bound_num_repairs=1)
            orig_md5 = pt.md5(p)
            if repaired is None:
                continue
            repaired_md5 = pt.md5(repaired)
            if orig_md5 == repaired_md5:
                continue
            passes[i].append(repaired_md5)
            num_remaining -= 1
            pbar.update(1)
        pbar.close()
    return passes


def test_repair_deterministic():
    rules = []
    seed = 42
    random_rule_sampler = get_rule_sampler(
        "mutation",
        None,
    )
    random_enumerator = get_tree_enumerator(
        "beam", random_rule_sampler, force_apply=True)
    random_repairer = PipelineRepairer(random_enumerator)
    print("Random-mutation")
    random_passes = get_repair_hashes(random_repairer, seed)
    assert len(random_passes[0]) > 0
    assert len(set(random_passes[0])) > 1
    assert random_passes[0] == random_passes[
        1], "random-mutation should be deterministic"

    # hack up some "fake" rules
    rules = []
    for _, lineage in random_enumerator.statistics.trace:
        for r in lineage:
            r._score_delta = np.random.random()
            rules.append(r)
    weighted_rule_sampler = get_rule_sampler("weighted", rules)
    weighted_enumerator = get_tree_enumerator(
        "beam",
        weighted_rule_sampler,
        force_apply=False,
    )
    weighted_repairer = PipelineRepairer(weighted_enumerator)
    print("Weighted-transducer")
    weighted_passes = get_repair_hashes(weighted_repairer, seed)
    assert len(weighted_passes[0]) > 0
    assert len(set(weighted_passes[0])) > 1
    assert weighted_passes[0] == weighted_passes[
        1], "weighted-transducer should be deterministic"
