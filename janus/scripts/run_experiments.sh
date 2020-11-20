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
MAX_TIME_MINS=5
CV=5
FIXED_SEED=42
SCORING_FUN="f1_macro"

NUM_TEST_PIPELINES=100
NUM_REPAIRED_PIPELINES=1
K=5
SEARCH_STRATEGIES="tpot random"
#REPAIR_STRATEGIES="weighted-transducer random-mutation rf-transducer"
REPAIR_STRATEGIES="random-mutation weighted-transducer"

if [[ ! -z "${TEST_CONFIG}" ]]
then
  echo "Running with TEST_CONFIG parameters (for quick test)"
  # setup a quick test configuration
  DATASETS="diabetes cars"
  NUM_TEST_PIPELINES=20
fi

# tsp params
NPROCS=10
tsp -S ${NPROCS}


echo "Running synthetic experiments"
for pipeline_source in ${SEARCH_STRATEGIES}
do
    for rule_source in ${SEARCH_STRATEGIES}
    do

        # skipping, not as interesting
        # I had originally thought...
        if [[ ${pipeline_source} != ${rule_source} ]]
        then
            continue
        fi


        for dataset in ${DATASETS}
        do
          # excludes rules extracted from $dataset
          rules_to_use=$(python scripts/utils.py \
            --cmd remove_contains \
            --arg1 ${RESULTS}/${rule_source}/*local-rules.pkl \
            --arg2 ${dataset})

            for repair_strategy in ${REPAIR_STRATEGIES}
            do
              output_dir="${RESULTS}/${pipeline_source}-pipelines-with-${rule_source}-rules/"
              mkdir -p ${output_dir}
              output_file="${output_dir}/${dataset}-synthetic-evaluation-${repair_strategy}.pkl"

              if [ -f ${output_file} ] && [ ${FORCE} -eq 0 ]
              then
                 echo "${output_file} exists and FORCE=0, so skipping"
                 echo "call with --force to execute regardless"
                 continue
              fi

              tsp python -m janus.evaluation.synthetic_evaluation \
                --rules ${rules_to_use} \
                --predefined_strategy ${repair_strategy} \
                --num_test_pipelines ${NUM_TEST_PIPELINES} \
                --test "${RESULTS}/${pipeline_source}/${dataset}-pipelines.pkl" \
                --idx_search "${RESULTS}/${pipeline_source}/${dataset}-pipelines.pkl-idx-search" \
                --bound_num_repaired_pipelines ${NUM_REPAIRED_PIPELINES} \
                --bound_k ${K} \
                --cv ${CV} \
                --scoring ${SCORING_FUN} \
                --random_state ${FIXED_SEED} \
                --output ${output_file}
            done
          done
      done
done

block-until-done-check-deadlocks $((${MAX_TIME_MINS} * 60)) $((${MAX_TIME_MINS} * 4 * 60))


echo "Running user-script experiments"
for search in ${SEARCH_STRATEGIES}
do
  output_dir="${RESULTS}/${search}-pipelines-with-${search}-rules/"
  mkdir -p ${output_dir}
  for repair_strategy in ${REPAIR_STRATEGIES}
  do
      output_file="${output_dir}/code-evaluation-${repair_strategy}.pkl"
      tsp python -m janus.evaluation.code_evaluation \
        --predefined_strategy ${repair_strategy} \
        --rules ${RESULTS}/${search}/*local-rules.pkl \
        --scripts janus/evaluation/user-scripts/*.py \
        --bound_num_repaired_pipelines ${NUM_REPAIRED_PIPELINES} \
        --scoring ${SCORING_FUN} \
        --bound_k ${K} \
        --cv ${CV} \
        --random_state ${FIXED_SEED} \
        --output ${output_file}
  done
done

block-until-done-check-deadlocks $((${MAX_TIME_MINS} * 60)) $((${MAX_TIME_MINS} * 4 * 60))



echo "Running random post-tree sampling experiment"
INIT_SEED=${FIXED_SEED}
for dataset in ${DATASETS}
do
    seed=${INIT_SEED}
    # offset seed for each example
    # avoid starting search same way
    INIT_SEED=$((${INIT_SEED} + 1))

    for search in ${SEARCH_STRATEGIES}
      do
        output_dir="${RESULTS}/${search}/random-post-sampler/"
        mkdir -p ${output_dir}

        tsp python -m janus.repair.tree_pairs \
          --input "${RESULTS}/${search}/${dataset}-pipelines.pkl" \
          --k 10 \
          --num_pre 200 \
          --num_post 50 \
          --output "${output_dir}/${dataset}-tree-pairs.pkl" \
          --seed ${seed} \
          --sample_method "random"
    done
done

if [[ ! -z "${TEST_CONFIG}" ]]
then
  block-until-done $((${MAX_TIME_MINS} * 60))
  echo "Skipping exact tree sampling when running with TEST_CONFIG"
  exit 0
fi


echo "Running exact post-tree sampling experiment"
INIT_SEED=${FIXED_SEED}
for dataset in ${DATASETS}
do
    seed=${INIT_SEED}
    # offset seed for each example
    # avoid starting search same way
    INIT_SEED=$((${INIT_SEED} + 1))

    for search in ${SEARCH_STRATEGIES}
      do
        output_dir="${RESULTS}/${search}/exact-post-sampler/"
        mkdir -p ${output_dir}

        tsp python -m janus.repair.tree_pairs \
          --input "${RESULTS}/${search}/${dataset}-pipelines.pkl" \
          --k 10 \
          --num_pre 200 \
          --num_post 50 \
          --output "${output_dir}/${dataset}-tree-pairs.pkl" \
          --seed ${seed} \
          --sample_method "exact"
    done
done

block-until-done $((${MAX_TIME_MINS} * 60))
