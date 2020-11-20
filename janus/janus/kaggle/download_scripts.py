#!/usr/env/bin python3
from argparse import ArgumentParser
import ast
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

from janus.kaggle.utils import CollectImports


def has_kaggle_cli():
    ret = subprocess.call(["kaggle", "--help"], stdout=subprocess.DEVNULL)
    return ret == 0


def download_kernel_paths(competition, max_num, output):
    cmd = """
    kaggle kernels list \
      --competition {competition} \
      --language python \
      --kernel-type all \
      --sort-by voteCount \
      --page-size {max_num} | \
      awk '$NF != 0' | \
      grep -v "Warning" | \
      grep "/" | \
      awk '{{print $1}}' > {output}
    """
    cmd = cmd.format(competition=competition, max_num=max_num, output=output)
    ret = subprocess.call(cmd, shell=True)
    return ret


def is_notebook(path):
    return path.endswith("ipynb")


def convert_notebook_to_script(path):
    # not use nbconvert b/c fails on a lot of kaggle notebooks for some reason
    with open(path, "r") as fin:
        nb = json.load(fin)
    cells = nb["cells"]
    cell_srcs = [c["source"] for c in cells if c["cell_type"] == "code"]
    code = "\n".join(cell_srcs)
    # remove ipython directives
    lines = [l for l in code.split("\n") if not l.startswith(("%", "!"))]
    clean_src = "\n".join(lines)
    new_path = os.path.splitext(path)[0] + ".py"
    with open(new_path, "w") as fout:
        fout.write(clean_src)
        fout.write("\n")

    return new_path


def compile_with(lang_ver, path):
    return subprocess.call([lang_ver, "-m", "py_compile", path])


def convert_to_py3(path):
    ret = subprocess.call(["2to3", "-w", path])
    if ret != 0:
        return None
    backup = path + ".bak"
    if os.path.exists(backup):
        os.remove(backup)
    return path


def imports_target_libs(script_path):
    target_libs = ["sklearn", "xgboost"]
    with open(script_path, "r") as fin:
        src = fin.read()
        imports = CollectImports().run(src)
        imports = [l.split(".")[0] for l in imports]
        return any(l in imports for l in target_libs)


def get_kernel_script(kernel_url, tmp_dir, output_path):
    curr_dir = os.getcwd()
    os.chdir(tmp_dir)

    ret = subprocess.call(["kaggle", "kernels", "pull", kernel_url, "--wp"])
    if ret != 0:
        print("Failed to pull", kernel_url)
        os.chdir(curr_dir)
        return None

    files = os.listdir(".")
    assert len(files) == 1
    file = files[0]

    if is_notebook(file):
        file = convert_notebook_to_script(file)
        if file is None:
            print("Failed converting notebook")
            os.chdir(curr_dir)
            return None

    status = compile_with("python3", file)
    if status != 0:
        print("Can't compile with python3, trying python2")
        if compile_with("python2", file) != 0:
            print("Failed to compile wiht python2")
            os.chdir(curr_dir)
            return None

        file = convert_to_py3(file)
        if file is None:
            print("Failed to convert to python3")
            os.chdir(curr_dir)
            return None

    if not imports_target_libs(file):
        print("Doesn't import a target lib")
        os.chdir(curr_dir)
        return None

    os.chdir(curr_dir)
    file_full_path = os.path.join(tmp_dir, file)
    os.rename(file_full_path, output_path)
    return output_path


def clean_tmp(tmp_dir):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        return
    else:
        shutil.rmtree(tmp_dir)
        return clean_tmp(tmp_dir)


def create_script_set(competition, max_num, output_dir):
    paths_file = os.path.join(output_dir, "paths.txt")
    download_kernel_paths(competition, max_num, paths_file)
    tmp_dir = os.path.join(output_dir, "tmp")
    clean_tmp(tmp_dir)
    collected = []
    ct = 0
    with open(paths_file, "r") as fin:
        for kernel_url in fin.readlines():
            kernel_url = kernel_url.strip()
            print("Pulling", kernel_url)
            output_path = os.path.join(output_dir, "script_{}.py".format(ct))
            result = get_kernel_script(kernel_url, tmp_dir, output_path)
            clean_tmp(tmp_dir)
            if result is not None:
                ct += 1
                collected.append({
                    "kernel_url": kernel_url,
                    "script_path": output_path
                })
    df_summary = pd.DataFrame(collected)
    df_summary.to_csv(os.path.join(output_dir, "sources.csv"), index=False)


def get_args():
    parser = ArgumentParser(
        description="Download kaggle scripts and munge them a bit")
    parser.add_argument(
        "--competition",
        type=str,
        choices=[
            "forest-cover-type-prediction",
            "ghouls-goblins-and-ghosts-boo",
            "otto-group-product-classification-challenge",
        ],
        help="Competition from which to download kernels",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        help="Maximum number of initial search results",
        default=100,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for munged scripts",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not has_kaggle_cli():
        print(
            "Must install the kaggle CLI, please see https://github.com/Kaggle/kaggle-api"
        )
        sys.exit(1)
    create_script_set(args.competition, args.max_num, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
