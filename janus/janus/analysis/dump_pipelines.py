#!/usr/bin/env python3
from argparse import ArgumentParser
import difflib
import json
import os
import subprocess

import pandas as pd
import tqdm

from janus.pipeline import pipeline_to_tree as pt
from janus.analysis import performance_analysis as pa



def get_str_diff(t1, t2):
    with open("t1.txt", "w") as fout:
        fout.write(t1)
    with open("t2.txt", "w") as fout:
        fout.write(t2)

    proc = subprocess.Popen(
        ["diff", "t1.txt", "t2.txt"],
        stdout=subprocess.PIPE,
    )
    diff_msg, _ = proc.communicate()
    os.remove("t1.txt")
    os.remove("t2.txt")
    return diff_msg.decode()


def get_pipelines_json(prepared_df):
    prepared_df["original_pipeline"] = [
        pt.to_pipeline(g) if not pd.isnull(g) else None
        for g in tqdm.tqdm(prepared_df["graph_orig"])
    ]
    prepared_df["repaired_pipeline"] = [
        pt.to_pipeline(g) if not pd.isnull(g) else None
        for g in tqdm.tqdm(prepared_df["graph_repaired"])
    ]

    prepared_df["original_tree"] = [
        pt.to_tree(p) if not pd.isnull(p) else None
        for p in tqdm.tqdm(prepared_df["original_pipeline"])
    ]
    prepared_df["repaired_tree"] = [
        pt.to_tree(p) if not pd.isnull(p) else None
        for p in tqdm.tqdm(prepared_df["repaired_pipeline"])
    ]

    prepared_df["original_json"] = [
        pt.to_json(t, payload_as_str=True) if not pd.isnull(t) else None
        for t in tqdm.tqdm(prepared_df["original_tree"])
    ]
    prepared_df["repaired_json"] = [
        pt.to_json(t, payload_as_str=True) if not pd.isnull(t) else None
        for t in tqdm.tqdm(prepared_df["repaired_tree"])
    ]

    prepared_df["original_text"] = [
        pt.to_text(t) if not pd.isnull(t) else None
        for t in tqdm.tqdm(prepared_df["original_tree"])
    ]
    prepared_df["repaired_text"] = [
        pt.to_text(t) if not pd.isnull(t) else None
        for t in tqdm.tqdm(prepared_df["repaired_tree"])
    ]

    prepared_df["text_diff"] = [
        get_str_diff(t1, t2) for t1, t2 in tqdm.tqdm(
            list(
                zip(prepared_df["original_text"],
                    prepared_df["repaired_text"])))
    ]

    prepared_df["original_score"] = prepared_df["mean_test_score_orig"]
    prepared_df["repaired_score"] = prepared_df["mean_test_score_repaired"]
    prepared_df["score_delta"] = prepared_df["repaired_score"] - prepared_df["original_score"]

    worse = prepared_df[prepared_df["hurt"]]
    same = prepared_df[(~prepared_df["hurt"]) & (~prepared_df["improved"])]
    better = prepared_df[prepared_df["improved"]]

    prepared_df = prepared_df[[
        "dataset",
        "original_json",
        "original_score",
        "repaired_json",
        "repaired_score",
        "improved",
        "hurt",
        "score_delta",
        "text_diff",
    ]]

    results = []
    for _, row in prepared_df.iterrows():
        results.append(dict(row))
    return results


def get_args():
    parser = ArgumentParser(description="Dump pipelines and repairs as json")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path to experiment output pickled files")
    parser.add_argument("--strategy", type=str, help="Strategy to dump")
    parser.add_argument("--output", type=str, help="Path to dump file")
    return parser.parse_args()


def main():
    args = get_args()
    args = get_args()
    dfs = []
    for path in args.input[:1]:
        df = pd.read_pickle(path)
        # fixing so older results....
        df["mean_test_score"] = df["mean_test_score"].astype(float)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    combined_df = combined_df[combined_df["strategy"] == args.strategy]
    prepared_df = pa.prepare_df(df, compute_dist=False)
    json_output = get_pipelines_json(prepared_df)
    with open(args.output, "w") as fout:
        json.dump(json_output, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
