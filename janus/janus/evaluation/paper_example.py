# #!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import difflib
import os
import sys

from bs4 import BeautifulSoup
import glob
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.stats.contingency_tables
import tqdm
import zss

from janus.pipeline import pipeline_to_tree as pt
from janus.analysis import performance_analysis as pa
from janus.analysis import rule_analysis as ra
from janus.repair.local_rules import (
    is_match_edit,
    edit_to_str,
    get_safe_label,
)
from janus.repair.local_rules import (
    ComponentInsert,
    ComponentRemove,
    ComponentUpdate,
    HyperparamRule,
    HyperparamRemove,
    HyperparamUpdate,
    get_safe_label,
    CorpusEntry,
    RuleCorpus,
)
from janus import utils
from janus.repair.tree_pairs import CorpusEntry, TreePairCorpus


def remove_diff_legend(src):
    parsed = BeautifulSoup(src)
    # second table is legend
    legend = parsed.find_all("table")[1]
    legend.extract()
    return str(parsed)


def show_diff_tree(orig, repaired, path=None):
    orig_str = pt.to_text(orig, indent_str="  ")
    repaired_str = pt.to_text(repaired, indent_str="  ")

    diff = difflib.HtmlDiff().make_file(
        orig_str.split("\n"),
        repaired_str.split("\n"),
        fromdesc="orig",
        todesc="repaired",
        context=True,
    )
    diff = remove_diff_legend(diff)
    if path is not None:
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(diff)
    return diff


def get_rules(paths, seed=42):
    rules = ra.load_rules(paths)
    sampler = ra.get_rule_sampler("weighted", rules, random_state=seed)
    sampler_rules = [r for group in sampler.rule_map.values() for r in group]
    df_sampler = ra.df_from_rules(sampler_rules)
    return df_sampler


def get_tree_emoji():
    return "\U0001F332"


def get_code_emoji():
    return "\U0001F4C4"


def add_ellipsis_row(df, format_numeric=False):
    df = df.copy()
    if format_numeric:
        col_types = df.dtypes
        is_numeric = col_types.map(lambda x: np.issubdtype(x, np.number))
        num_cols = col_types[is_numeric].index.values
        for c in num_cols:
            df[c] = df[c].map(lambda x: "{:.2f}".format(x))
    row = pd.DataFrame([{c: "..." for c in df.columns}])
    return pd.concat((df, row), axis=0)


def show_trace(improved):
    entries = [(get_code_emoji(), s) for s in improved["mean_test_score_orig"]]
    df = pd.DataFrame(entries, columns=["pipeline", "performance"])
    df = df.head(4)
    return add_ellipsis_row(df, format_numeric=True)


def show_tree_pairs(tree_pairs_corpus, threshold=10):
    # sat our distance threshold
    entries = [e for e in tree_pairs_corpus.entries if e.distance < threshold]
    # dominating pair
    entries = [
        e for e in entries if not pd.isnull(e.pre.external_score)
        and not pd.isnull(e.post.external_score)
        and e.pre.external_score < e.post.external_score
    ]

    records = []
    for e in entries:
        record = {
            "tree_1": get_tree_emoji(),
            "tree_2": get_tree_emoji(),
            "distance": e.distance,
            "score_delta": (e.post.external_score - e.pre.external_score)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.groupby("distance").head(1)
    df = df.head(3)
    return add_ellipsis_row(df, format_numeric=True)


def show_edit_ops(tree_pairs_corpus, seed=42):
    # non match edits
    edits = [
        op for e in tree_pairs_corpus.entries for op in e.edits
        if op.type != zss.Operation.match
    ]
    edits = [{"edit_type": e.type, "edit_str": edit_to_str(e)} for e in edits]

    df = pd.DataFrame(edits)
    df = df.sample(frac=1.0, replace=False, random_state=seed)
    df = df.groupby("edit_type").head(1)
    df = df[["edit_str"]].rename(columns={"edit_str": "edit"})
    return add_ellipsis_row(df)


def rule_name_for_paper(r_str):
    names = {
        "ComponentInsert": "CInsert",
        "ComponentRemove": "CRemove",
        "ComponentUpdate": "CUpdate",
        "HyperparamRemove": "HRemove",
        "HyperparamUpdate": "HUpdate",
    }
    for orig, replacement in names.items():
        r_str = r_str.replace(orig, replacement)
    return r_str


def show_rules(df_rules, seed=42):
    # to latex html table
    # show rules of each type
    # for ComponentInsert pick ones that don't
    # just insert a stacking estimator
    df_rules = df_rules.copy()
    cond_insert = (
        df_rules["type_str"] == "ComponentInsert"
    ) & (df_rules["post_label"].map(
        lambda x: isinstance(x, str) and not x.endswith("StackingEstimator")))
    others = df_rules["type_str"] != "ComponentInsert"
    df_rules = df_rules[cond_insert | others]
    df_rules = df_rules.sample(frac=1, replace=False, random_state=seed)
    df_rules = df_rules.groupby("type_str").head(1)
    df_rules["rule_str"] = df_rules["rule"].map(lambda x: x.as_str())
    df_rules["rule_str"] = df_rules["rule_str"].map(rule_name_for_paper)
    rule_strs = df_rules["rule_str"]
    df = rule_strs.to_frame(name="LSR")
    return add_ellipsis_row(df)


def get_args():
    parser = ArgumentParser(description="Extended system diagram")
    parser.add_argument("--input_dir", type=str, help="Folder for $RESULTS")
    parser.add_argument("--output_dir", type=str, help="Folder for $ANALYSIS")
    parser.add_argument("--seed", type=int, help="RNG seed", default=42)
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_pickle(
        os.path.join(
            args.input_dir,
            "tpot-pipelines-with-tpot-rules/car-evaluation-synthetic-evaluation-janus.pkl"
        ))
    prepared_df = pa.prepare_df(df, compute_dist=True)
    improved = prepared_df[prepared_df["improved"]]
    improved = improved.sort_values("distance", ascending=False)

    tree_pairs = pd.read_pickle(
        os.path.join(args.input_dir, "tpot/car-evaluation-tree-pairs.pkl"))

    rules = get_rules(
        glob.glob(os.path.join(args.input_dir, "tpot/*-local-rules.pkl")),
        seed=args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    trace_df = show_trace(improved)
    trace_df.to_html(
        os.path.join(args.output_dir, "trace.html"),
        index=False,
        float_format="{:.2f}".format,
    )

    tree_pairs_df = show_tree_pairs(tree_pairs)
    tree_pairs_df.to_html(
        os.path.join(args.output_dir, "tree-pairs.html"),
        index=False,
        float_format="{:.2f}".format,
    )

    edits_df = show_edit_ops(tree_pairs, seed=args.seed)
    edits_df.to_html(os.path.join(args.output_dir, "edits.html"), index=False)

    lsr_df = show_rules(rules, seed=args.seed)
    lsr_df.to_html(
        os.path.join(args.output_dir, "lsr.html"),
        index=False,
    )

    # Janus applied
    min_score_diff = 0.05
    improved = improved[improved.score_diff >= min_score_diff]
    improved = improved.sort_values("score_diff", ascending=True)
    ix = 0
    orig = improved.iloc[ix].graph_orig
    repaired = improved.iloc[ix].graph_repaired

    print("Orig graph")
    print(pt.to_text(orig))

    print("-----------")

    print("Repaired graph")
    print(pt.to_text(repaired))

    print("Score improvement: ", improved.iloc[ix].score_diff)
    print("From: ", improved.iloc[ix].mean_test_score_orig)
    print("To: ", improved.iloc[ix].mean_test_score_repaired)
    show_diff_tree(orig, repaired, os.path.join(args.output_dir, "diff.html"))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
