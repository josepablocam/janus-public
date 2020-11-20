#!/usr/bin/env bash

source scripts/run_setup.sh
source scripts/utils.sh

DATASETS="Hill_Valley_without_noise "
DATASETS+="Hill_Valley_with_noise "
DATASETS+="breast-cancer-wisconsin "
DATASETS+="car-evaluation "
DATASETS+="glass "
DATASETS+="ionosphere "
DATASETS+="spambase "
DATASETS+="wine-quality-red "
DATASETS+="wine-quality-white "

# evaluation params
MAX_TIME_MINS=60
TSP_CHECK_EVERY_SECS=$((5 * 60))
TSP_DEADLOCK_SECS=$((20 * 60))
HOLD_OUT_SIZE=0.5
TEST_SIZE=0.2
INIT_SEED=42
FIXED_SEED=${INIT_SEED}
N_JOBS=1
SCORING_FUN="f1_macro"

if [[ ! -z "${TEST_CONFIG}" ]]
then
  echo "Running with TEST_CONFIG parameters (for quick test)"
  # setup a quick test configuration
  DATASETS="diabetes cars"
  # evaluation params
  MAX_TIME_MINS=2
  TSP_CHECK_EVERY_SECS=$((${MAX_TIME_MINS} * 60))
  TSP_DEADLOCK_SECS=$((${MAX_TIME_MINS} * 5 * 60))
fi

SEARCH_STRATEGIES="tpot random"

# tsp params
NPROCS=10
tsp -S ${NPROCS}


echo "Collecting pipelines"
INIT_SEED=${FIXED_SEED}
for dataset in ${DATASETS}
do
    seed=${INIT_SEED}
    # offset seed for each example
    # avoid starting search same way
    INIT_SEED=$((${INIT_SEED} + 1))

    for search in ${SEARCH_STRATEGIES}
    do
        mkdir -p "${RESULTS}/${search}"
        tsp python -m janus.pipeline.collect_pipelines \
            --search ${search} \
            --dataset ${dataset} \
            --hold_out_size ${HOLD_OUT_SIZE} \
            --test_size ${TEST_SIZE} \
            --max_time_mins ${MAX_TIME_MINS} \
            --random_state ${seed} \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${RESULTS}/${search}/${dataset}-pipelines.pkl"
    done
done

block-until-done-check-deadlocks ${TSP_CHECK_EVERY_SECS} ${TSP_DEADLOCK_SECS}

# Construct pairs of pre/post tree
echo "Constructing pre/post tree pairs"
INIT_SEED=${FIXED_SEED}
for dataset in ${DATASETS}
do
    seed=${INIT_SEED}
    # offset seed for each example
    # avoid starting search same way
    INIT_SEED=$((${INIT_SEED} + 1))

    for search in ${SEARCH_STRATEGIES}
      do
      tsp python -m janus.repair.tree_pairs \
          --input "${RESULTS}/${search}/${dataset}-pipelines.pkl" \
          --k 10 \
          --num_pre 200 \
          --num_post 50 \
          --output "${RESULTS}/${search}/${dataset}-tree-pairs.pkl" \
          --seed ${seed} \
          --sample_method "approximate"
    done
done

block-until-done ${TSP_CHECK_EVERY_SECS}


echo "Extracting local edit rules"
INIT_SEED=${FIXED_SEED}
# Extract local edit rules from paired trees
for dataset in ${DATASETS}
do
    seed=${INIT_SEED}
    # offset seed for each example
    # avoid starting search same way
    INIT_SEED=$((${INIT_SEED} + 1))

    for search in ${SEARCH_STRATEGIES}
    do
      # outputs
      tsp python -m janus.repair.local_rules \
          --input "${RESULTS}/${search}/${dataset}-tree-pairs.pkl" \
          --max_edit_distance 10 \
          --output "${RESULTS}/${search}/${dataset}-local-rules.pkl"
    done
done


block-until-done ${TSP_CHECK_EVERY_SECS}
