#!/usr/bin/env python3
from argparse import ArgumentParser

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from janus.lift.extract_pipeline import PipelineLifter
from janus import utils


def get_target_remapping(dataset_path):
    data_name = os.path.basename(dataset_path)
    target_name = {
        "forest-cover.csv": "Cover_Type",
        "ghouls.csv": "type",
        "otto.csv": "target",
    }
    return {"target": target_name[data_name]}


def add_fake_id_column(df, dataset_path):
    if dataset_path.endswith("forest-cover.csv"):
        name = "Id"
    else:
        name = "id"
    if name not in df.columns:
        df[name] = np.arange(0, df.shape[0])
    return df


def prepare_dataset(dataset_path, max_size):
    df = pd.read_csv(dataset_path)
    sampled_path = dataset_path + "-sampled"
    if max_size is not None and max_size < df.shape[0]:
        df = df.sample(n=max_size, replace=False)
    # rename target column
    target_remap = get_target_remapping(dataset_path)
    df = df.rename(columns=target_remap)
    df = add_fake_id_column(df, dataset_path)
    df.to_csv(sampled_path, index=False)
    return sampled_path


def create_kaggle_dir_structure(tmp,
                                script_path,
                                dataset_path,
                                max_size=None,
                                random_state=42):
    data_dir = os.path.join(tmp, "input")
    src_dir = os.path.join(tmp, "src")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    script_name = os.path.basename(script_path)
    new_script_path = os.path.join(src_dir, script_name)
    shutil.copy(script_path, new_script_path)

    # create downsampled dataset if needed
    utils.set_seed(random_state)
    dataset_path = prepare_dataset(dataset_path, max_size)

    # symlink the dataset as train.csv *and* test.csv
    # we are only interested in getting the pipeline
    # not any performance coming from this
    train_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_path):
        os.remove(train_path)
    os.symlink(dataset_path, train_path)

    test_path = os.path.join(data_dir, "test.csv")
    if os.path.exists(test_path):
        os.remove(test_path)
    os.symlink(dataset_path, test_path)

    return new_script_path


def get_args():
    parser = ArgumentParser(description="Run kaggle script and lift pipelines")
    parser.add_argument("--script", type=str, help="Path to script")
    parser.add_argument("--tmp", type=str, help="Temporary directory")
    parser.add_argument(
        "--keep_tmp", action="store_true", help="Don't delete temporary")
    parser.add_argument("--dataset", type=str, help="Path to csv dataset")
    parser.add_argument(
        "--output", type=str, help="Path to store lifted pipelines (pickled)")
    parser.add_argument(
        "--max_size",
        type=int,
        help="Max number of rows in dataset (downsample if needed)",
        default=1000,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="RNG seed",
        default=42,
    )
    return parser.parse_args()


def main():
    args = get_args()
    output_path = os.path.abspath(args.output)

    if args.tmp is None:
        tmp = tempfile.TemporaryDirectory()
        tmp_dir_path = tmp.name
    else:
        tmp_dir_path = args.tmp

    if not os.path.exists(tmp_dir_path):
        os.makedirs(tmp_dir_path)

    new_script_path = create_kaggle_dir_structure(
        tmp_dir_path,
        args.script,
        os.path.abspath(args.dataset),
        max_size=args.max_size,
        random_state=args.random_state,
    )

    orig_dir = os.getcwd()
    script_dir = os.path.dirname(new_script_path)
    script_name = os.path.basename(new_script_path)

    os.chdir(script_dir)

    print("Running {}".format(script_name))

    lifted = PipelineLifter(script_name)
    if not lifted.failed:
        print("Dumping pipelines for", args.script)
        lifted.dump(output_path)
    else:
        print("Failed extracting pipelines for", args.script)

    os.chdir(orig_dir)

    if args.keep_tmp:
        return

    if args.tmp is None:
        tmp.cleanup()
    else:
        shutil.rmtree(args.tmp)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
