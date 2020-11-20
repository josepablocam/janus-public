#!/usr/bin/env bash
source scripts/run_setup.sh
source scripts/utils.sh

COMPETITIONS[0]="forest-cover-type-prediction"
COMPETITIONS[1]="ghouls-goblins-and-ghosts-boo"
COMPETITIONS[2]="otto-group-product-classification-challenge"

COMPETITION_DATASETS[0]="forest-cover.csv"
COMPETITION_DATASETS[1]="ghouls.csv"
COMPETITION_DATASETS[2]="otto.csv"

if [[ ! -z "${TEST_CONFIG}" ]]
then
    echo "Running with TEST_CONFIG parameters (for quick test)"
    MAX_NUM_SCRIPTS=2
    MAX_TIME=$((5 * 60))
else
    MAX_NUM_SCRIPTS=100
    MAX_TIME=$((60 * 60))
fi

# max number of rows in downsampled dataset
MAX_SIZE=1000
# tsp params
NPROCS=10
tsp -S ${NPROCS}

SEED=42


for i in ${!COMPETITIONS[@]}
do
  comp=${COMPETITIONS[$i]}
  echo "Running Kaggle scripts for ${comp}"

  dataset="${DATASETS_FOLDER}/${COMPETITION_DATASETS[$i]}"

  comp_dir="${KAGGLE_FOLDER}/${comp}"
  output_dir="${KAGGLE_FOLDER}/pipelines/${comp}"
  mkdir -p ${output_dir}

  for script in $(ls ${comp_dir}/*.py | head -n ${MAX_NUM_SCRIPTS})
  do
      output_path="${output_dir}/$(basename ${script}).pkl"

      tsp timeout ${MAX_TIME} \
        python -m janus.kaggle.collect_pipelines \
        --script ${script} \
        --dataset ${dataset} \
        --output ${output_path} \
        --random_state ${SEED}

      # advance seed
      SEED=$((${SEED} + 1))
  done
done

block-until-done $((${MAX_TIME} * 60))
