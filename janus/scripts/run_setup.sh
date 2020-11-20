#!/usr/bin/env bash
# make sure can call `conda activate etc`
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]
then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    # assume we're on rhino
    source "/raid/$(whoami)/miniconda3/etc/profile.d/conda.sh"
fi
conda activate pipeline-repair-env

export DATA="../datasets/"

if [[ ! -z "${TEST_CONFIG}" ]]
then
    export RESULTS="../results-test/"
else
    export RESULTS="../results/"
fi


export ANALYSIS="../analysis/"
export DATASETS_FOLDER="../datasets/"
export KAGGLE_FOLDER="../kaggle/"

mkdir -p ${RESULTS}
mkdir -p ${ANALYSIS}
mkdir -p ${KAGGLE_FOLDER}

export DOCKER_USER="janus-user"
export DOCKER_CONTAINER_NAME="janus-container"
