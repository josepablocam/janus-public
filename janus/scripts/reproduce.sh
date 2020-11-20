#!/usr/bin/env bash
FORCE=0
while [[ "$#" -gt 0 ]]
do
    case $1 in
        --force) FORCE=1; shift;;
        *) echo "Unknown parameter: $1";exit 1;;
    esac
done

source scripts/run_setup.sh
bash scripts/run_collect_data.sh


if [[ ${FORCE} -eq 1 ]]
then
    bash scripts/run_experiments.sh --force
else
    bash scripts/run_experiments.sh
fi

bash scripts/run_analysis.sh
bash scripts/run_paper_example.sh
