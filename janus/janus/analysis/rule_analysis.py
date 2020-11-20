#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import os
import pickle

import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
from janus.repair.tree_enumerator import EnumerationStatistics
from janus.repair.rule_sampler import get_rule_sampler

from janus.analysis import utils


def df_from_rules(rules):
    entries = []
    for r in rules:
        if isinstance(r, tuple):
            strategy = r[1]
            r = r[0]
        else:
            strategy = "mined"
        record = {
            "rule": r,
            "strategy": strategy,
            "type": type(r),
        }
        if isinstance(r, HyperparamRule):
            record["parent_label"] = get_safe_label(r.pre.parent)
            record["pre_label"] = r.pre.payload[0]
            record["pre_value"] = r.pre.payload[1]
            if isinstance(r, HyperparamRemove):
                record["post_value"] = None
            else:
                record["post_value"] = r.post.payload[1]
                try:
                    record["post_value_greater"] = record[
                        "post_value"] > record["pre_value"]
                except TypeError:
                    record["post_value_greater"] = np.nan
        else:
            record["pre_label"] = get_safe_label(r.pre)
            record["post_label"] = get_safe_label(r.post)
        entries.append(record)
    df = pd.DataFrame(entries)
    df["type_str"] = df["type"].map(lambda x: x.__name__)
    return df


def is_hyperrule(df):
    return df["type"].map(lambda t: issubclass(t, HyperparamRule))


def get_hyperrules(df):
    return df[is_hyperrule(df)]


def get_comprules(df):
    return df[~is_hyperrule(df)]


def get_simplename(s):
    return s.split(".")[-1]


def grid_plot(df, x):
    g = sns.FacetGrid(data=df, col="strategy")

    def ordered_countplot(x, **kwargs):
        order = pd.Series(x).value_counts().index
        sns.countplot(x=x, order=order, **kwargs)

    g.map(ordered_countplot, x)
    return g


def single_plot(df, y, **kwargs):
    fig, ax = plt.subplots(1)
    # alphabetical order
    order = list(sorted(df[y].unique()))
    sns.countplot(data=df, y=y, order=order, ax=ax, **kwargs)
    return ax


def freq_plot(df, x, use_single_plot=False, **kwargs):
    if df.shape[0] == 0:
        return None
    if (not use_single_plot) and len(df["strategy"].unique()) > 1:
        ax = grid_plot(df, x)
    else:
        ax = single_plot(df, x, **kwargs)
    rotate_xticks(ax)
    return ax


def set_xlabel(obj, lab):
    if obj is None:
        return None
    if isinstance(obj, sns.FacetGrid):
        obj.set_xlabels(lab)
    else:
        obj.set_xlabel(lab)


def set_ylabel(obj, lab):
    if obj is None:
        return None
    if isinstance(obj, sns.FacetGrid):
        obj.set_ylabels(lab)
    else:
        obj.set_ylabel(lab)


def set_title(obj, title):
    pass
    # TODO: need to fix titling...
    # if obj is None:
    #     return None
    # if isinstance(obj, sns.FacetGrid):
    #     obj.fig.suptitle(title)
    #     obj.fig.subplots_adjust(top=.9)
    # else:
    #     obj.set_title(title)


def rotate_xticks(obj):
    if obj is None:
        return None
    if isinstance(obj, sns.FacetGrid):
        for ax in obj.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
    else:
        plt.xticks(rotation=90)


def savefig(obj, path):
    if obj is None:
        return
    if isinstance(obj, sns.FacetGrid):
        obj.savefig(path)
    else:
        obj.get_figure().savefig(path)


def histogram_rule_type(df):
    df = df.copy()
    kwargs = {"hue": "strategy", "dodge": True, "orient": "v"}
    ax = freq_plot(df, "type_str", use_single_plot=True, **kwargs)
    set_xlabel(ax, "Rule Type")
    set_ylabel(ax, "Count")
    set_title(ax, "Rule Distribution")
    plt.legend(loc="best", title=None, ncol=2)
    plt.tight_layout()
    return ax


def histogram_hyperrule_component(df):
    df = get_hyperrules(df).copy()
    df["parent_label"] = df["parent_label"].map(get_simplename)
    ax = freq_plot(df, "parent_label")
    set_xlabel(ax, "Component")
    set_ylabel(ax, "Count")
    set_title(ax, "Hyperparameter Rule Components")
    plt.tight_layout()
    return ax


def histogram_comprule_component(df):
    df = get_comprules(df).copy()
    df["pre_label"] = df["pre_label"].map(get_simplename)
    ax = freq_plot(df, "pre_label")
    set_xlabel(ax, "Component")
    set_ylabel(ax, "Count")
    set_title(ax, "Component Rule Applications")
    plt.tight_layout()
    return ax


def ggflow_plot(df, x, y, path):
    # TODO: get ggflow up and running
    tmpfile = utils.get_tmpfile(suffix=".csv")
    df.to_csv(tmpfile.name)

    outputfile = utils.get_tmpfile(suffix=".pdf", delete=False)
    R_code = """
    library(ggplot2)

    dat <- read.csv({file}, stringsAsFactors=FALSE)
    plot <- ggplot(
        dat,
        aes(
            y=freq,
            axis1=pre,
            axis2=post,
        )
    ) + geom_alluvium()
    plot <- plot + labs(
        x=""
    )

    """
    R_code = R_code.format(file=tmpfile.name, output=outputfile.name)
    utils.run_R(R_code)
    return outputfile.name


def flow_comprules(df):
    df = get_comprules(df)
    pass


def flow_hyperrules(df):
    pass


def get_enumerator_strategy(obj, path):
    try:
        strategy = obj.strategy
        if strategy is not None:
            return obj.strategy
        else:
            raise AttributeError()
    except AttributeError:
        prefix = os.path.basename(path).split(".")[0]
        parts = prefix.split("-")
        ix = parts.index("evaluation") + 1
        return "-".join(parts[ix:])


def load_rules(paths):
    rules = []
    for p in paths:
        with open(p, "rb") as fin:
            obj = pickle.load(fin)
            if isinstance(obj, RuleCorpus):
                rules.extend(obj.rules)
            elif isinstance(obj, EnumerationStatistics):
                for _, lineage in obj.trace:
                    strategy = get_enumerator_strategy(obj, p)
                    annotated_lineage = [(r, strategy) for r in lineage]
                    rules.extend(annotated_lineage)
            else:
                raise TypeError("Unknown type to extract rules: " +
                                str(type(obj)))
    return rules


def rule_name_for_paper(r):
    names = {
        "ComponentInsert": "CInsert",
        "ComponentRemove": "CRemove",
        "ComponentUpdate": "CUpdate",
        "HyperparamRemove": "HRemove",
        "HyperparamUpdate": "HUpdate",
    }
    return names.get(r)


def get_args():
    parser = ArgumentParser(description="Visualize rule distribution")
    parser.add_argument("--input", type=str, nargs="+", help="Input files")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--rule_sampler",
        type=str,
        choices=["weighted"],
        help=
        "Summarize using a rule sampler and compare to originally mined numbers"
    )
    parser.add_argument("--seed", type=int, help="RNG seed", default=42)
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rules = load_rules(args.input)

    df = df_from_rules(rules)
    df["type_str"] = df["type_str"].map(rule_name_for_paper)
    print("Total count of raw rules", df.shape[0])

    df_sampler = None
    if args.rule_sampler is not None:
        sampler = get_rule_sampler(
            args.rule_sampler, rules, random_state=args.seed)
        sampler_rules = [
            r for group in sampler.rule_map.values() for r in group
        ]
        df_sampler = df_from_rules(sampler_rules)
        print("Total count of summarized rules", df_sampler.shape[0])

    ax = histogram_rule_type(df)
    savefig(ax, os.path.join(args.output_dir, "rule_type.pdf"))

    if df_sampler is not None:
        df_sampler["strategy"] = "summarized"
        ax = histogram_rule_type(df_sampler)
        # remove y axis label (shared with other plot)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        savefig(ax, os.path.join(args.output_dir, "rule_type_summarized.pdf"))

    ax = histogram_hyperrule_component(df)
    savefig(ax, os.path.join(args.output_dir, "hyperrules.pdf"))

    ax = histogram_comprule_component(df)
    savefig(ax, os.path.join(args.output_dir, "comprules.pdf"))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
