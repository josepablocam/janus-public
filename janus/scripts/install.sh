#!/usr/bin/env bash
set -ex

function install_conda_if_needed() {
    if ! command -v conda &> /dev/null
      then
        curl https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -L --output miniconda.sh\
            && bash miniconda.sh -b \
            && rm -f miniconda.sh

            export PATH=miniconda3/bin/:${PATH}
            source ~/miniconda3/etc/profile.d/conda.sh
    fi
}


install_conda_if_needed
conda env create -f environment.yml

source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh

git submodule update --init --recursive || echo "If running in dockerfile, must git submodule update before building"
conda activate pipeline-repair-env

# install tpot
pushd tpot
pip install -e .
popd

# install python-pl
pushd python-pl
pip install -e .
popd


if [[ "$#" -eq 1 ]] && [[ "${1}" == --kaggle ]]
then
  # install packages specific to kaggle scripts we want to run
  # don't pollute broader environment in general, only if prompted with
  # --kaggle
  # install one at a time so failures don't stop installing rest
    source scripts/run_setup.sh
    cat ${KAGGLE_FOLDER}/requirements.txt | grep -v "#" | xargs -I {} pip install {}
fi

# explicit exit code, since xargs can return 123
exit 0
