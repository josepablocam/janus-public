#!/usr/bin/env python3
from collections import Counter
import os
import time
import sys

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
import stopit
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.tree_enumerator import TreeEnumerator
from janus import mp_utils
from janus import utils

# TODO: ad hoc for now
if os.environ.get("DEBUG", None) is None:
    DEFAULT_TIMEOUT_PER_REPAIR = 60
    DEFAULT_TIMEOUT_TOTAL = 60
else:
    DEFAULT_TIMEOUT_PER_REPAIR = int(1e6)
    DEFAULT_TIMEOUT_TOTAL = int(1e6)


class RepairStatistics(object):
    def __init__(self):
        self.info = {
            "exceptions": [],
            "successes": [],
        }
        self.strategy = None

    def record_failure(self, err):
        err_info = (utils.get_exception_name(err), utils.get_exception_tb(err))
        self.info["exceptions"].append(err_info)

    def record_success(self, ct_tries, time):
        self.info["successes"].append((ct_tries, time))


class PipelineRepairer(object):
    def __init__(self, enumerator, total_time=DEFAULT_TIMEOUT_TOTAL):
        assert isinstance(enumerator, TreeEnumerator)
        self.enumerator = enumerator
        self.total_time = total_time
        self.debug_data = []
        self.statistics = RepairStatistics()

    def repair(
            self,
            pipeline,
            X,
            y,
            bound_k=3,
            bound_num_repairs=10,
            scoring="f1_macro",
            cv=3,
            random_state=42,
            verbosity=0,
    ):
        self.debug_data = []
        cv_splitter = StratifiedKFold(
            cv,
            random_state=random_state,
            shuffle=True,
        )

        orig_tree = pt.to_tree(pipeline)
        tried = set([])
        tried.add(pt.to_hashable_json(orig_tree))

        repair_ct = 0
        best_candidate = None
        best_score = None
        pbar = tqdm.tqdm(total=bound_num_repairs)

        time_budget_left = self.total_time
        tree_gen = self.enumerator.enumerate(orig_tree, bound_k)

        # attempted pipelines before one executes without failure
        ct_tries = 0
        time_last_pipeline = time.time()

        while repair_ct < bound_num_repairs and time_budget_left > 0:
            if verbosity > 0:
                sys.stdout.write(
                    "Time left:{:.2f}(s)\r".format(time_budget_left))
                sys.stdout.flush()
            iter_start_time = time.time()

            # generate candidate trees in separate thread
            # so we can time out
            with stopit.ThreadingTimeout(time_budget_left) as ctx:
                try:
                    candidate_tree = next(tree_gen)
                except StopIteration:
                    time_budget_left = 0
                    break

            if ctx.state == ctx.TIMED_OUT:
                time_budget_left = 0
                break

            if candidate_tree is None:
                time_budget_left -= (time.time() - iter_start_time)
                continue

            candidate_hash = pt.to_hashable_json(candidate_tree)

            if candidate_hash in tried:
                time_budget_left -= (time.time() - iter_start_time)
                continue

            try:
                compiled_candidate = pt.to_pipeline(candidate_tree)
            except Exception as err:
                ct_tries += 1
                self.statistics.record_failure(err)
                time_budget_left -= (time.time() - iter_start_time)
                continue

            tried.add(candidate_hash)
            try:
                repair_results = mp_utils.run(
                    DEFAULT_TIMEOUT_PER_REPAIR,
                    cross_validate,
                    compiled_candidate,
                    X,
                    y,
                    cv=cv_splitter,
                    scoring=scoring,
                    return_estimator=True,
                    return_train_score=True,
                )
                self.debug_data.append(compiled_candidate)
                now = time.time()
                time_taken = now - time_last_pipeline
                time_last_pipeline = now
                self.statistics.record_success(ct_tries, time_taken)
                time_budget_left -= (time.time() - iter_start_time)
            # based on
            # https://github.com/josepablocam/ams/blob/master/core/search.py#L274
            except (
                    TimeoutError,
                    ValueError,
                    TypeError,
                    ZeroDivisionError,
                    IndexError,
                    AttributeError,
                    MemoryError,
                    ImportError,
                    mp_utils.TimeoutError,
                    mp_utils.mp.pool.MaybeEncodingError,
            ) as err:
                ct_tries += 1
                self.statistics.record_failure(err)
                time_budget_left -= (time.time() - iter_start_time)
                continue

            repair_ct += 1
            candidate_score = np.mean(repair_results["test_score"])
            if best_candidate is None or candidate_score > best_score:
                best_candidate = compiled_candidate
                best_score = candidate_score
            pbar.update(1)
        pbar.close()
        return best_candidate
