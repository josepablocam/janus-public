#!/usr/bin/env python3

import subprocess
import tempfile


def get_tmpfile(suffix=None, delete=True):
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=delete,
    )
    return tmp


def run_R(src):
    tmp = get_tmpfile(suffix=".R")
    tmp.write(src)
    tmp.flush()
    status = subprocess.call(["Rscript", tmp.name])
    assert status == 0
    tmp.close()
