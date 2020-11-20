#!/usr/bin/env bash

source scripts/run_setup.sh

python -m janus.kaggle.create_requirements \
  --input ${KAGGLE_FOLDER}/*/*.py \
  --filter \
  > "${KAGGLE_FOLDER}/requirements.txt"
