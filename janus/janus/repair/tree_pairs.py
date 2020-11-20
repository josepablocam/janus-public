#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import heapq
import pickle
import random
import re
import time
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus import utils


def pipeline_tokenizer(s):
  tokens = re.findall(r"\w+", s)
  # add structural markers (), {}
  extra = [c for c in s if c in "(){}"]
  return tokens + extra


class CorpusEntry(object):
    def __init__(self, pre, post, distance, edits):
        self.key = pre["json"]
        self.pre = pre
        self.post = post
        self.distance = distance
        self.edits = edits

    def __lt__(self, other):
        # heapq maintains minheap, but we want
        # maxheap, so just use negative distance
        return (-self.distance) < (-other.distance)


class TreePairCorpus(object):
    def __init__(self, entries, compute_time, sample_method):
        self.entries = entries
        self.compute_time = compute_time
        self.sample_method = sample_method

    def __iter__(self):
        return iter(self.entries)


class PairedCorpusBuilder(object):
    def __init__(self, k):
        self.k = k
        # maintain a maxheap for each `pre` in the pair
        self.pairs = defaultdict(lambda: [])

    def push(self, entry):
        pairs_for_pre = self.pairs[entry.key]
        if len(pairs_for_pre) < self.k:
            heapq.heappush(pairs_for_pre, entry)
        else:
            most_distant_entry = heapq.heappop(pairs_for_pre)
            if most_distant_entry.distance > entry.distance:
                # replace it with the new entry
                # which is closer to pre
                heapq.heappush(pairs_for_pre, entry)
            else:
                heapq.heappush(pairs_for_pre, most_distant_entry)

    def get_entries(self):
        entries = [entry for group in self.pairs.values() for entry in group]
        return sorted(entries, key=lambda x: x.distance)


class ApproximatePostSampler(object):
    def __init__(self, pipelines_df):
        str_reps = pipelines_df["obj_key"].map(str)
        # treat string representation of pipeline
        # as text and tokenize with counts
        # this accoutns to some extent for structure with things
        # like count of parens, components etc
        mat = CountVectorizer(tokenizer=pipeline_tokenizer).fit_transform(str_reps)
        # approximate distance using this text encoding
        approx_dist_mat = pairwise_distances(mat, mat)
        highest = approx_dist_mat.max(axis=0).reshape(-1, 1)
        # set diagonal to infinity, since never want to sample self
        np.fill_diagonal(approx_dist_mat, np.inf)
        # lowest distances
        lowest = approx_dist_mat.min(axis=0).reshape(-1, 1)
        approx_norm_mat = (approx_dist_mat - lowest) / (highest - lowest)
        self.mat = approx_norm_mat

    def sample(self, pre_ix, k, candidates):
        weights = self.mat[pre_ix, candidates]
        candidates = np.array(candidates)
        return candidates[np.argsort(weights)][:k]


class ExactPostSampler(object):
    def __init__(self, pipelines_df):
        n = pipelines_df.shape[0]
        self.pipelines_df = pipelines_df
        self.dist_mat = np.repeat(np.inf, n * n).reshape((n, n))
        self.computed_mat = np.zeros((n, n)).astype(bool)
        self.edits = {}

    def get_distance_and_edits(self, pre_ix, post_ix):
        # distance is symmetric, but edit ops are not
        if self.computed_mat[pre_ix, post_ix]:
            distance = self.dist_mat[pre_ix, post_ix]
            edits = self.edits[(pre_ix, post_ix)]
            return distance, edits
        else:
            pipeline1 = self.pipelines_df.iloc[pre_ix]["obj_graph"]
            pipeline2 = self.pipelines_df.iloc[post_ix]["obj_graph"]
            distance, edits = pt.tree_edit_distance(
                pipeline1,
                pipeline2,
                return_operations=True,
            )
            self.computed_mat[pre_ix, post_ix] = True
            self.dist_mat[pre_ix, post_ix] = distance
            self.edits[(pre_ix, post_ix)] = edits
            return distance, edits

    def sample(self, pre_ix, k, candidates):
        dists_and_edits = []
        for post_ix in tqdm.tqdm(candidates):
            d, e = self.get_distance_and_edits(pre_ix, post_ix)
            dists_and_edits.append((d, e))
        ordered = sorted(
            zip(candidates, dists_and_edits), key=lambda x: x[1][0])
        return [(post_ix, d, e) for (post_ix, (d, e)) in ordered[:k]]


class RandomPostSampler(ApproximatePostSampler):
    def sample(self, pre_ix, k, candidates):
        return np.random.choice(
            candidates,
            size=min(k, len(candidates)),
            replace=False,
        )


def build_paired_corpus(
        pipelines_df,
        num_pre,
        num_post,
        k,
        sample_method="approximate",
):
    start_time = time.time()
    builder = PairedCorpusBuilder(k)
    # make sure only unique pipelines
    pipelines_df["json"] = pipelines_df["obj_graph"].map(pt.to_hashable_json)
    column_types = pipelines_df.dtypes
    float_cols = column_types[column_types == float].index.values
    other_cols = column_types[column_types != float].index.values
    agg_ops = {col: np.mean for col in float_cols}
    agg_ops.update({col: (lambda x: x.values[0]) for col in other_cols})
    agg_ops.pop("json")
    pipelines_df = pipelines_df.groupby("json").agg(agg_ops)
    pipelines_df = pipelines_df.reset_index()
    pipelines_df["failed"] = pipelines_df["external_score"].isna()
    success_ixs = np.where(~pipelines_df["failed"])[0].tolist()

    print("Building post-sampler of type: {}".format(sample_method))
    if sample_method == "random":
        post_sampler = RandomPostSampler(pipelines_df)
    elif sample_method == "approximate":
        post_sampler = ApproximatePostSampler(pipelines_df)
    elif sample_method == "exact":
        post_sampler = ExactPostSampler(pipelines_df)
    else:
        raise Exception("Unknown sample method: {}".format(sample_method))

    n = pipelines_df.shape[0]
    ixs = np.arange(0, n)
    pre_ixs = ixs.tolist()

    if num_pre is not None:
        random.shuffle(pre_ixs)
        pre_ixs = pre_ixs[:num_pre]

    for pre_ix in tqdm.tqdm(pre_ixs):
        pre = pipelines_df.iloc[pre_ix]

        if pre.failed:
            # any succeeding pipeline can be post if pre is failure
            post_ixs = list(success_ixs)
        else:
            # only higher scores can be post
            higher_score = pipelines_df["external_score"] > pre.external_score

            success = ~pipelines_df["failed"]
            post_ixs = (ixs[success & higher_score]).tolist()

        if num_post is not None:
            sampled_results = post_sampler.sample(pre_ix, num_post, post_ixs)

        for res in tqdm.tqdm(sampled_results):
            if sample_method == "exact":
                post_ix, distance, edits = res
            else:
                post_ix, distance, edits = res, None, None

            if post_ix == pre_ix:
                continue
            post = pipelines_df.iloc[post_ix]
            assert not post.failed, "Post-tree can never be a failure"
            if distance is None:
                # only compute if needed
                distance, edits = pt.tree_edit_distance(
                    pre.obj_graph,
                    post.obj_graph,
                    return_operations=True,
                )
            entry = CorpusEntry(pre, post, distance, edits)
            builder.push(entry)

    end_time = time.time()
    entries = builder.get_entries()
    for entry in entries:
        # add parent/sibling info
        entry.pre.obj_graph.annotate()
        entry.post.obj_graph.annotate()

    corpus = TreePairCorpus(
        entries,
        compute_time=end_time - start_time,
        sample_method=sample_method,
    )
    return corpus


def get_args():
    parser = ArgumentParser(
        description="Create tree pairs from search trace dataframe")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to pickled search trace dataframe",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Keep k post-trees with lowest distance for each pre-tree",
    )
    parser.add_argument(
        "--num_pre",
        type=int,
        help="Number of pre-trees to sample",
    )
    parser.add_argument(
        "--num_post",
        type=int,
        help="Number of post-trees to sample per pre-tree",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save down pickled tree pairs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed to set",
        default=42,
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        choices=["exact", "approximate", "random"],
        help=
        "Post-tree sampler approach (exact/random are only for evaluation)",
        default="approximate",
    )
    return parser.parse_args()


def main():
    args = get_args()
    pipelines_df = pd.read_pickle(args.input)
    utils.set_seed(args.seed)
    pairs = build_paired_corpus(
        pipelines_df,
        args.num_pre,
        args.num_post,
        args.k,
        sample_method=args.sample_method,
    )
    with open(args.output, "wb") as fout:
        pickle.dump(pairs, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # noqa
        import pdb
        pdb.post_mortem()
        sys.exit(1)
