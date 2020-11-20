#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import os
import pickle

import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from janus.repair.tree_pairs import CorpusEntry, TreePairCorpus


def create_distance_df(corpus):
    records = []
    for entry in corpus:
        rec = {"distance": entry.distance, "method": corpus.sample_method}
        records.append(rec)
    return pd.DataFrame(records)


def plot_distance_ecdf(df):
    fig, ax = plt.subplots(1)
    sns.ecdfplot(data=df, x="distance", hue="method", ax=ax)
    return ax


def get_args():
    parser = ArgumentParser(description="Compare post-tree sampling methods")
    parser.add_argument(
        "--input", type=str, nargs="+", help="TreePairCorpus files")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for analysis results")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dist_dfs = []
    compute_times = defaultdict(lambda: [])

    for path in args.input:
        with open(path, "rb") as fin:
            corpus = pickle.load(fin)
            dist_df = create_distance_df(corpus)
            dist_dfs.append(dist_df)
            compute_times[corpus.sample_method].append(corpus.compute_time)

    combined_dist_df = pd.concat(dist_dfs, axis=0)
    dist_ecdf = plot_distance_ecdf(combined_dist_df)
    dist_ecdf.get_figure().savefig(
        os.path.join(args.output_dir, "distance_ecdf.pdf"))

    summary_df = combined_dist_df.groupby("method")["distance"].agg(
        ["mean", "max", "std"])
    summary_df = summary_df.reset_index()

    compute_times_info = {
        k: (np.mean(v), np.std(v))
        for k, v in compute_times.items()
    }
    summary_df["mean_compute_time_str"] = [
        "{:.2f}(+/- {:.2f})".format(*compute_times_info[m])
        for m in summary_df["method"]
    ]
    summary_df["mean_distance_str"] = [
        "{:.2f}(+/- {:.2f})".format(m, d)
        for m, d in zip(summary_df["mean"], summary_df["std"])
    ]
    summary_df = summary_df[[
        "method", "mean_distance_str", "mean_compute_time_str"
    ]]
    summary_df = summary_df.rename(
        columns={
            "method": "Sampling method",
            "mean_distance_str": "Mean (SD) Distance",
            "mean_compute_time_str": "Mean (SD) Time (s)"
        })
    summary_df.to_latex(
        os.path.join(args.output_dir, "summary.tex"), index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
