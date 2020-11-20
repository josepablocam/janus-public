#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import datetime
import dateutil.parser
import subprocess
import sys

# if there is no output to stdout for
# 20 minutes, we say it is likely deadlocked
# so we can kill process and restart
STDOUT_TIMEOUT_DEADLOCK = 20 * 60


def get_real_pid(tsp_pid):
    proc = subprocess.run("tsp -p {}".format(tsp_pid),
                          shell=True,
                          stdout=subprocess.PIPE)
    if proc.returncode == 0:
        return int(proc.stdout.decode("utf-8").strip())
    else:
        # if not running, no PID
        return None


def run_tsp_clear():
    subprocess.run("tsp -C", shell=True)


def get_timestamp_stdout_modified(file_path):
    proc = subprocess.run("stat {}".format(file_path),
                          shell=True,
                          stdout=subprocess.PIPE)
    for line in proc.stdout.decode("utf-8").strip().split("\n"):
        if line.startswith("Modify"):
            tokens = line.split()
            date_str = tokens[1]
            time_str = tokens[2]
            ts = "{} {}".format(date_str, time_str)
            return dateutil.parser.parse(ts)


def get_tsp_info():
    proc = subprocess.run("tsp", shell=True, stdout=subprocess.PIPE)
    lines = proc.stdout.decode("utf-8").split("\n")
    parsed = []
    for line in lines[1:]:
        if len(line.strip()) == 0:
            continue
        tokens = line.split()
        entry = {}
        entry["status"] = tokens[1]
        if entry["status"] == "running":
            entry["pid"] = get_real_pid(tokens[0])
        else:
            entry["pid"] = None
        entry["stdout"] = tokens[2]
        ix = 3
        if entry["status"] == "finished":
            entry["exitcode"] = int(tokens[ix])
            ix += 1
            entry["times"] = tokens[ix]
            ix += 1
        entry["command"] = " ".join(tokens[ix:])
        parsed.append(entry)
    return parsed


def get_likely_tsp_deadlocks(
    entries,
    max_diff_seconds,
):
    entries = [
        e for e in entries if e["status"] == "running" and e["pid"] is not None
    ]
    deadlocks = []
    for e in entries:
        ts = get_timestamp_stdout_modified(e["stdout"])
        time_diff_secs = (datetime.now() - ts).seconds
        if time_diff_secs > max_diff_seconds:
            deadlocks.append(e)
    return deadlocks


def kill_and_requeue_tsp(entries):
    print("{} possible deadlocks".format(len(entries)))
    print("Killing and requeueing")

    for e in entries:
        pid = e["pid"]
        subprocess.run("kill {}".format(pid), shell=True)
    for e in entries:
        # reschedule
        cmd = "tsp {}".format(e["command"])
        subprocess.run(cmd, shell=True)


def monitor_tsp(timeout_seconds, _arg2):
    if timeout_seconds is not None:
        timeout_seconds = int(timeout_seconds[0])
    else:
        timeout_seconds = STDOUT_TIMEOUT_DEADLOCK
    assert _arg2 is None
    entries = get_tsp_info()
    finished = [e for e in entries if e["status"] == "finished"]
    if all([e["exitcode"] == 0 for e in finished]):
        run_tsp_clear()
    deadlocks = get_likely_tsp_deadlocks(
        entries,
        max_diff_seconds=timeout_seconds,
    )
    if len(deadlocks) > 0:
        kill_and_requeue_tsp(deadlocks)
    else:
        print("No deadlocks so far")


def remove_contains(entries, remove):
    new_entries = [e for e in entries if not any(r in e for r in remove)]
    return new_entries


def get_args():
    parser = ArgumentParser(description="Utils for scripting")
    parser.add_argument(
        "--cmd",
        type=str,
        help="Command to run",
        choices=["remove_contains", "monitor_tsp"],
    )
    parser.add_argument("--arg1", type=str, nargs="+", help="Arg1")
    parser.add_argument("--arg2", type=str, nargs="+", help="Arg2")
    return parser.parse_args()


def main():
    args = get_args()
    cmd = None
    if args.cmd == "remove_contains":
        cmd = remove_contains
    elif args.cmd == "monitor_tsp":
        cmd = monitor_tsp
    else:
        raise Exception("No such command")

    result = cmd(args.arg1, args.arg2)
    if result is not None:
        print(" ".join(result), file=sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
