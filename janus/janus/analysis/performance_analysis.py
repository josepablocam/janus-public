#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys

import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.stats.contingency_tables
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.repair.local_rules import (
    is_match_edit,
    edit_to_str,
)
from janus import utils


def add_timestamp_percentile(df, percentiles, percentiles_labels):
    unique_ts = df.groupby(["dataset", "id"]).head(1)
    percentile_map = {}
    for d in unique_ts["dataset"].unique():
        unique_ts_d = unique_ts[unique_ts["dataset"] == d]
        unique_ts_d = unique_ts_d.sort_values("timestamp", ascending=True)
        percents = pd.qcut(
            unique_ts_d["timestamp"], percentiles, labels=percentiles_labels)
        for i, p in zip(unique_ts_d["id"], percents):
            percentile_map[(d, i)] = p

    return [percentile_map[(d, i)] for d, i in zip(df["dataset"], df["id"])]


def prepare_df(df, compute_dist=False):
    df_orig = df[df["type"] == "orig"]
    df_orig = df_orig[~df_orig["failed"]]

    # make sure we only consider dataset/id where we have the orig
    # for all strategies
    unique_strategies = df["strategy"].unique()
    n_strategies = len(unique_strategies)
    strategy_cts = df_orig.groupby(
        ["dataset", "id"])["strategy"].agg(lambda x: len(set(x)))
    strategy_cts = strategy_cts.to_frame(name="strategy_cts").reset_index()
    df_orig = pd.merge(df_orig, strategy_cts, how="left", on=["dataset", "id"])
    df_orig = df_orig[df_orig["strategy_cts"] == n_strategies]

    df_repaired = df[df["type"] == "repair"]
    df_repaired = df_repaired[~df_repaired["failed"]]

    df_repaired = df_repaired.sort_values("mean_test_score", ascending=False)

    # there should only be one secore per dataset/id/strategy
    assert df_repaired.groupby(["dataset", "strategy", "id"]).size().max() == 1
    df_orig = df_orig[[
        "dataset",
        "strategy",
        "id",
        "mean_test_score",
        "graph",
        "timestamp",
    ]]
    df_repaired = df_repaired[[
        "dataset", "strategy", "id", "mean_test_score", "graph"
    ]]

    df_combined = pd.merge(
        df_orig,
        df_repaired,
        how="left",
        on=["dataset", "strategy", "id"],
        suffixes=("_orig", "_repaired"),
    )
    if compute_dist:
        dist = [
            None if pd.isnull(repaired) else pt.tree_edit_distance(
                orig, repaired) for orig, repaired in tqdm.tqdm(
                    list(
                        zip(df_combined["graph_orig"],
                            df_combined["graph_repaired"])))
        ]
    else:
        dist = np.nan
    df_combined["distance"] = dist

    # assign "row" to timestamp-based quartile
    df_combined["ts_quartile"] = add_timestamp_percentile(
        df_combined,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"],
    )
    df_combined["ts_decile"] = add_timestamp_percentile(
        df_combined, np.arange(0, 1.1, 0.1),
        (lambda x: ["{:.1f}-{:.1f}".format(i, j) for i, j in zip(x, x[1:])])(
            np.arange(0, 1.1, 0.1)))

    df_combined["score_diff"] = df_combined[
        "mean_test_score_repaired"] - df_combined["mean_test_score_orig"]
    df_combined["improved"] = (df_combined["score_diff"] >
                               0) & (~pd.isnull(df_combined["score_diff"]))
    df_combined["improved_int"] = df_combined["improved"].astype(int)
    df_combined["has_repair"] = ~pd.isnull(
        df_combined["mean_test_score_repaired"])
    df_combined["dummy"] = 1
    return df_combined


def stat_by_strategy(df, col, groupcols=None, op="mean"):
    if groupcols is None:
        groupcols = ["dataset", "strategy"]
    assert "strategy" in groupcols
    stat_df = df.groupby(groupcols)[col].agg(op)
    stat_df = stat_df.to_frame(name=col)
    stat_df = stat_df.reset_index()
    index_cols = list(groupcols)
    index_cols.remove("strategy")
    pv_stat_df = pd.pivot_table(
        data=stat_df, index=index_cols, columns="strategy", values=col)
    pv_stat_df = pv_stat_df.reset_index()
    pv_stat_df.columns.name = None
    return pv_stat_df


def summarize_df(df):
    df = df.copy()
    # now append version with "overall" (i.e. agg) ts_quartile
    df_overall = df.copy()
    df_overall["ts_quartile"] = "overall"

    df = pd.concat((df, df_overall), axis=0).reset_index(drop=True)
    groupcols = ["dataset", "ts_quartile", "strategy"]

    print("Stat: Number of pipelines in experiments")
    print(stat_by_strategy(df, "dummy", groupcols=groupcols, op="sum"), "\n")

    print("Stat: Fraction w/ available 'repair'")
    print(stat_by_strategy(df, "has_repair", groupcols=groupcols, op="mean"),
          "\n")

    print("Stat: Fraction improved")
    print(stat_by_strategy(df, "improved", groupcols=groupcols, op="mean"),
          "\n")

    print("Stat: Total number improved")
    print(stat_by_strategy(
        df[df["ts_quartile"] == "overall"],
        "improved_int",
        groupcols=["dataset", "strategy"],
        op="sum"), "\n")

    print("Stat: Mean score diff")
    print(stat_by_strategy(df, "score_diff", groupcols=groupcols, op="mean"),
          "\n")

    print("Stat: Mean score diff (if improvement)")
    print(stat_by_strategy(
        df[df["score_diff"] > 0], "score_diff", groupcols=groupcols,
        op="mean"), "\n")

    df = df.sort_values("score_diff", ascending=False)
    best_df = df.groupby(["dataset", "ts_quartile", "id"]).head(1)
    print(
        "Stat: Number of dataset/pipeline where a strategy gets largest score improvement"
    )
    print(stat_by_strategy(best_df, "dummy", groupcols=groupcols, op="sum"))

    print("Stat: Mean distance (if improvement)")
    print(stat_by_strategy(
        df[df["score_diff"] > 0], "distance", groupcols=groupcols, op="mean"),
          "\n")


def plot_fraction_repaired(df):
    fig, ax = plt.subplots(1)
    sns.pointplot(
        data=df,
        x="improved",
        y="dataset",
        hue="strategy",
        estimator=np.mean,
        linestyles=["None"] * len(df["strategy"].unique()),
        dodge=True,
        ci=95,
        ax=ax,
    )
    ax.set_xlabel("Fraction of Pipelines Improved")
    ax.set_ylabel("Dataset")
    ax.set_xlim(0.0, 0.65)
    plt.legend(loc="center right", bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)
    plt.tight_layout()
    return ax


def fraction_repaired_over_time(df):
    fig, ax = plt.subplots(1)
    sns.pointplot(
        data=df,
        x="ts_decile",
        y="improved",
        hue="strategy",
        estimator=np.mean,
        dodge=True,
        ci=95,
        ax=ax,
    )
    plt.xticks(rotation=90)
    ax.set_xlabel("Timestamp Decile")
    ax.set_ylabel("Fraction of Pipelines Improved")
    plt.legend(loc="best", title=None, ncol=2)
    plt.tight_layout()
    return ax


def plot_score_improvement(df):
    all_strategies = df["strategy"].unique()
    n_all_strategies = len(all_strategies)
    df = df[df["improved"]]
    check = df.groupby(["dataset",
                        "id"])["strategy"].agg(lambda x: len(set(x)))
    check = check.to_frame(name="num_strategies").reset_index()
    check["drop"] = check["num_strategies"] < n_all_strategies
    df = pd.merge(df, check, how="left", on=["dataset", "id"])
    df["drop"] = df["drop"].fillna(True)
    df = df[~df["drop"]]
    fig, ax = plt.subplots(1)
    # sns.barplot(
    #     data=df,
    #     x="dataset",
    #     y="score_diff",
    #     hue="strategy",
    #     ci=95,
    #     ax=ax,
    # )
    sns.pointplot(
        data=df,
        x="score_diff",
        y="dataset",
        hue="strategy",
        linestyles=["None"] * len(df["strategy"].unique()),
        estimator=np.mean,
        dodge=True,
        ci=95,
        ax=ax,
    )
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Score Improvement\n(both improved)")
    plt.legend(loc="center right", bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)
    plt.tight_layout()
    return ax


def plot_score_improvement_st_distance(df):
    all_strategies = df["strategy"].unique()
    n_all_strategies = len(all_strategies)

    # only improvements
    df = df[df["improved"]]

    # now only consider cases where the random-mutation
    # had an edit distance <= the corresponding edit distance
    # for the janus repair
    df_janus = df[df["strategy"] == "janus"]
    df_janus = df_janus[["dataset", "id", "distance"]]
    df_janus = df_janus.rename(columns={"distance": "distance_compare"})
    assert df_janus.shape[0] > 0
    df = pd.merge(df, df_janus, how="left", on=["dataset", "id"])
    df = df[df["distance"] <= df["distance_compare"]]

    # both janus and random-mutation have improvement and
    # sat the edit distance constraint mentioned above
    check = df.groupby(["dataset",
                        "id"])["strategy"].agg(lambda x: len(set(x)))
    check = check.to_frame(name="num_strategies").reset_index()
    check["drop"] = check["num_strategies"] < n_all_strategies
    df = pd.merge(df, check, how="left", on=["dataset", "id"])
    df["drop"] = df["drop"].fillna(True)
    df = df[~df["drop"]]

    fig, ax = plt.subplots(1)
    sns.barplot(
        data=df,
        x="dataset",
        y="score_diff",
        hue="strategy",
        ci=95,
        ax=ax,
    )
    plt.xticks(rotation=90)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(
        "Score Improvement\n(both improved and\nsatisfy distance threshold)")
    plt.legend(loc="best", title=None, ncol=2)
    plt.tight_layout()
    return ax


def plot_fraction_repaired_st_distance(df):
    # now only consider cases where the random-mutation
    # had an edit distance <= the corresponding edit distance
    # for the janus repair
    df_janus = df[df["strategy"] == "janus"]
    df_janus = df_janus[["dataset", "id", "distance"]]
    df_janus = df_janus.rename(columns={"distance": "distance_compare"})
    assert df_janus.shape[0] > 0
    df = pd.merge(df, df_janus, how="left", on=["dataset", "id"])
    df = df[df["distance"] <= df["distance_compare"]]

    fig, ax = plt.subplots(1)
    sns.pointplot(
        data=df,
        x="dataset",
        y="improved",
        hue="strategy",
        estimator=np.mean,
        linestyles=["None"] * len(df["strategy"].unique()),
        dodge=True,
        ci=95,
        ax=ax,
    )
    plt.xticks(rotation=90)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Fraction of Pipelines Improved\n(subject to distance)")
    plt.legend(loc="best", title=None, ncol=2)
    plt.tight_layout()
    return ax


def plot_cdf_score_diff(df):
    df = df.copy()
    # don't count pipelines that were originally broken
    df = df[~pd.isnull(df["mean_test_score_orig"])]
    # but for cases where the repaired is broken
    # we say that it gets score 0.0
    df["mean_test_score_repaired"] = df["mean_test_score_repaired"].fillna(0.0)
    df["score_diff"] = df["mean_test_score_repaired"] - df[
        "mean_test_score_orig"]
    assert not pd.isnull(df["score_diff"]).any()
    fig, ax = plt.subplots(1)
    sns.ecdfplot(data=df, x="score_diff", hue="strategy", ax=ax)
    ax.set_xlabel("Pipeline score change")
    ax.set_ylabel("Empirical CDF")
    plt.tight_layout()
    return ax


def plot_dist_score_diff(df):
    df = df.copy()
    # don't count pipelines that were originally broken
    df = df[~pd.isnull(df["mean_test_score_orig"])]
    # but for cases where the repaired is broken
    # we say that it gets score 0.0
    df["mean_test_score_repaired"] = df["mean_test_score_repaired"].fillna(0.0)
    df["score_diff"] = df["mean_test_score_repaired"] - df[
        "mean_test_score_orig"]
    assert not pd.isnull(df["score_diff"]).any()
    fig, ax = plt.subplots(1)

    check = df.groupby(["dataset", "id"])["improved"].min()
    check = check.to_frame(name="drop").reset_index()
    df = pd.merge(df, check, how="left", on=["dataset", "id"])
    df["drop"] = df["drop"].fillna(True)
    df = df[~df["drop"]]

    sns.histplot(data=df, x="score_diff", hue="strategy", ax=ax)
    ax.set_xlabel("Pipeline score change")
    ax.set_ylabel("Distribution")
    plt.legend(loc="best", title=None, ncol=2)
    plt.tight_layout()
    return ax


def plot_cdf_distance(df):
    df = df[df["improved"]]
    print("Fraction of repair with distance > 10")
    df = df.copy()
    df["distance_over_10"] = df["distance"] > 10.0
    print(df.groupby("strategy")["distance_over_10"].mean())
    fig, ax = plt.subplots(1)
    sns.ecdfplot(data=df, x="distance", hue="strategy", ax=ax)
    ax.set_xlabel("Edit distance\n(improved pipelines)")
    ax.set_ylabel("Empirical CDF")
    plt.tight_layout()
    return ax


def stat_test_count_improved(
        df,
        strategies,
        num_comparisons=None,
        random_state=42,
):
    assert len(strategies) == 2
    df = df[df["strategy"].isin(strategies)]
    strategies_present = df["strategy"].unique()
    if len(strategies_present) == 1:
        print("Single strategy", strategies_present[0])
        print("Can't compute mcnemar")
        return None, None

    pv = pd.pivot_table(
        df, index=["dataset", "id"], columns="strategy",
        values="improved").reset_index()
    pvg = pv.groupby(strategies).size()
    pvg = pvg.to_frame(name="ct").reset_index()
    pvg_pv = pd.pivot_table(
        pvg,
        index=strategies[0],
        columns=strategies[1],
        values="ct",
    )
    # non-parametric for paired tests
    utils.set_seed(random_state)
    if pd.isnull(pvg_pv.values.flatten()).any():
        print("McNemar can't handle nans")
        return None, None

    cont_table = pvg_pv.values
    if cont_table.shape != (2, 2):
        print("McNemar requires well formed contingency table")
        return None, None

    obj = statsmodels.stats.contingency_tables.mcnemar(cont_table)
    stat = obj.statistic
    p_value = obj.pvalue
    if num_comparisons is not None:
        p_value = p_value * num_comparisons
    return stat, p_value


def stat_test_score_diff_improvements(df,
                                      strategies,
                                      num_comparisons=None,
                                      random_state=42):
    assert len(strategies) == 2
    df = df[df["strategy"].isin(strategies) & df["improved"]]
    strategies_present = df["strategy"].unique()
    if len(strategies_present) == 1:
        print("Single strategy", strategies_present[0])
        print("Can't compute paired t-test")
        return None, None

    pv_df = pd.pivot_table(
        df,
        index=["dataset", "id"],
        columns="strategy",
        values="score_diff",
    ).reset_index()
    no_missing = (~pd.isnull(pv_df[strategies[0]])) & (
        ~pd.isnull(pv_df[strategies[1]]))
    pv_df = pv_df[no_missing]

    scores_0 = pv_df[strategies[0]]
    scores_1 = pv_df[strategies[1]]
    utils.set_seed(random_state)
    stat, p_value = scipy.stats.ttest_rel(scores_0, scores_1)
    if num_comparisons is not None:
        p_value = p_value * num_comparisons
    return stat, p_value


def count_runtime_repair_exceptions(combined_df):
    df = combined_df[combined_df["type"] == "repair"]
    df = df.copy()
    df["exceptions"] = df["repairer_statistics"].map(
        lambda x: x.info["exceptions"])
    df["num_exceptions"] = df["exceptions"].map(len)
    df_exc = df.groupby("strategy")["num_exceptions"].sum()
    df_exc = df_exc.to_frame(name="Runtime Exceptions")
    df_exc = df_exc.reset_index()
    df_exc.columns = [c.capitalize() for c in df_exc.columns]
    return df_exc


def sample_examples(df, strategy, sample_n=10, random_state=42):
    df = df.copy()
    df = df[df["strategy"] == strategy]
    print("Strategy:", strategy)
    print("Most improved examples:")
    df = df[~pd.isnull(df["graph_repaired"])]
    df = df[df["score_diff"] > 0]
    df = df.sort_values("score_diff", ascending=False)
    df = df.head(sample_n)
    for _, row in df.iterrows():
        print("Dataset:", row.dataset)
        print("Score: {} -> {}".format(row.mean_test_score_orig,
                                       row.mean_test_score_repaired))
        print("Distance:", row.distance)

        _, edits = pt.tree_edit_distance(
            row.graph_orig,
            row.graph_repaired,
            return_operations=True,
        )
        print("Edits:")
        for e in edits:
            if not is_match_edit(e):
                print(edit_to_str(e))

        print("Orig pipeline:")
        print(pt.to_json(row.graph_orig))

        print("New pipeline:")
        print(pt.to_json(row.graph_repaired))

        print("\n\n")


def get_args():
    parser = ArgumentParser(description="Simple analysis")
    parser.add_argument(
        "--input", type=str, nargs="+", help="Experiment output files")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Strategies to include (all if None)",
    )
    parser.add_argument(
        "--compute_distance",
        action="store_true",
        help="Compute tree edit distance for all repairs",
    )
    parser.add_argument(
        "--num_comparisons",
        type=int,
        help="Number of comparisons for Bonferroni adjustment",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        help="Sample pipelines to show more detail",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="RNG seed",
        default=42,
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="String mapping <raw>:<desired-label>")
    return parser.parse_args()


def main():
    args = get_args()
    dfs = []
    for path in args.input:
        df = pd.read_pickle(path)
        # fixing so older results....
        df["mean_test_score"] = df["mean_test_score"].astype(float)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    combined_df["failed"] = combined_df["mean_test_score"].isna()
    # combined_df["timedout"] = combined_df["timedout"].fillna(False)
    if args.strategies is not None:
        combined_df = combined_df[combined_df["strategy"].isin(
            args.strategies)]
        combined_df = combined_df.reset_index(drop=True)

    if len(combined_df["type"].unique()) == 1:
        print("No repair types in: ", ",".join(args.input))
        sys.exit(0)

    prepared_df = prepare_df(combined_df, args.compute_distance)

    if args.labels is not None:
        mapped_labels = [l.split(":") for l in args.labels]
        assert all(len(p) == 2 for p in mapped_labels)
        mapped_labels = dict(mapped_labels)
        prepared_df["strategy"] = prepared_df["strategy"].map(mapped_labels)
        combined_df["strategy"] = combined_df["strategy"].map(mapped_labels)
        args.strategies = [mapped_labels[s] for s in args.strategies]

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 150)

    summarize_df(prepared_df)

    if args.sample_n is not None:
        print("Sampled pipelines")
        strategies = prepared_df["strategy"].unique()
        for s in strategies:
            sample_examples(prepared_df, s, args.sample_n, args.random_state)

    if args.strategies is not None:
        print("McNemar Paired Test (number of improvements)")
        stat, p_value = stat_test_count_improved(
            prepared_df,
            args.strategies,
            num_comparisons=args.num_comparisons,
            random_state=args.random_state,
        )
        print("Stat: {}, p-value: {}".format(stat, p_value))

        print("Paired t-test (Score diff for improvements)")
        stat, p_value = stat_test_score_diff_improvements(
            prepared_df,
            args.strategies,
            num_comparisons=args.num_comparisons,
        )
        print("Stat: {}, p-value: {}".format(stat, p_value))

    if args.output_dir is None:
        return 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ax = plot_fraction_repaired(prepared_df)
    ax.get_figure().savefig(
        os.path.join(args.output_dir, "fraction_repaired.pdf"))

    ax = fraction_repaired_over_time(prepared_df)
    ax.get_figure().savefig(
        os.path.join(args.output_dir, "fraction_repaired_over_time.pdf"))

    try:
        ax = plot_score_improvement(prepared_df)
        ax.get_figure().savefig(
            os.path.join(args.output_dir, "score_improvement.pdf"))
    except:
        print("Skipped score_improvement.pdf")
        pass

    try:
        ax = plot_score_improvement_st_distance(prepared_df)
        ax.get_figure().savefig(
            os.path.join(args.output_dir, "score_improvement_st_distance.pdf"))
    except:
        print("Skipped score_improvement_st_distance.pdf")
        pass

    if args.compute_distance:
        ax = plot_fraction_repaired_st_distance(prepared_df)
        ax.get_figure().savefig(
            os.path.join(args.output_dir, "fraction_repaired_st_distance.pdf"))

    ax = plot_cdf_score_diff(prepared_df)
    ax.get_figure().savefig(
        os.path.join(args.output_dir, "score_diff_cdf.pdf"))

    ax = plot_dist_score_diff(prepared_df)
    ax.get_figure().savefig(
        os.path.join(args.output_dir, "score_diff_distribution.pdf"))

    if args.compute_distance:
        ax = plot_cdf_distance(prepared_df)
        ax.get_figure().savefig(
            os.path.join(args.output_dir, "distance_cdf.pdf"))

    runtime_exc = count_runtime_repair_exceptions(combined_df)
    print(runtime_exc)
    runtime_exc.to_latex(
        os.path.join(args.output_dir, "runtime_repair_exceptions.tex"),
        index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
