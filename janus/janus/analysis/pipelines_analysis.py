#!/usr/bin/env python3
from argparse import ArgumentParser
import sys

import pandas as pd


def count_pipelines(df):
    return df.shape[0]


def get_args():
    parser = ArgumentParser(
        description="Basic stats over pipelines in automl search trace")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path to pickled search trace dataframe",
    )
    return parser.parse_args()


def main():
    args = get_args()
    acc = []
    for path in args.input:
        df = pd.read_pickle(path)
        acc.append(df)
    pipelines_df = pd.concat(acc, axis=0)
    print("Number of pipelines")
    print(count_pipelines(pipelines_df))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # noqa
        import pdb
        pdb.post_mortem()
        sys.exit(1)
