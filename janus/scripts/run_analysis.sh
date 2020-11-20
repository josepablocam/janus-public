#!/usr/bin/env bash
source scripts/run_setup.sh
source scripts/utils.sh

if [[ ! -z "${TEST_CONFIG}" ]]
then
  export RESULTS="../results-test/"
  export ANALYSIS="../analysis-test/"
  mkdir -p ${ANALYSIS}
fi


SEARCH_STRATEGIES="tpot random"
REPAIR_STRATEGIES="random-mutation weighted-transducer"
REPAIR_LABELS="random-mutation:random-mutation weighted-transducer:janus"
SEED=42

for experiment_dir in $(find ${RESULTS} -type d -iname *with*)
do
    analysis_dir="${ANALYSIS}/$(basename ${experiment_dir})/synthetic/"
    mkdir  -p ${analysis_dir}

    output_file="${analysis_dir}/performance_analysis.output"
    echo "Collecting ${output_file}"
    python -m janus.analysis.performance_analysis \
      --input ${experiment_dir}/*synthetic-evaluation*.pkl  \
      --strategies ${REPAIR_STRATEGIES} \
      --labels ${REPAIR_LABELS} \
      --output_dir ${analysis_dir} \
      --num_comparisons $(echo $REPAIR_STRATEGIES | wc -w) \
      --compute_distance \
     | tee ${output_file}


    echo "Producing rules applied analysis: ${experiment_dir}"
    python -m janus.analysis.rule_analysis \
      --input ${experiment_dir}/*synthetic-evaluation*-enumerator-statistics \
      --output_dir ${analysis_dir}


    analysis_dir="${ANALYSIS}/$(basename ${experiment_dir})/user-scripts/"
    mkdir -p ${analysis_dir}
    output_file="${analysis_dir}/performance_analysis.output"
    echo "Collecting ${output_file}"
    python -m janus.analysis.performance_analysis \
      --input ${experiment_dir}/*code-evaluation*.pkl  \
      --strategies ${REPAIR_STRATEGIES} \
      --labels ${REPAIR_LABELS} \
      --compute_distance \
      --output_dir ${analysis_dir} \
      | tee ${output_file}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}"
  mkdir -p ${analysis_dir}

  echo "Producing rules extracted analysis: ${search}"
  python -m janus.analysis.rule_analysis \
    --input ${RESULTS}/${search}/*-local-rules.pkl \
    --output_dir ${analysis_dir} \
    --rule_sampler weighted \
    --seed ${SEED}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}/tree-sampling/"
  mkdir -p ${analysis_dir}

  echo "Tree sampling analysis: ${search}"
  python -m janus.analysis.tree_pairs_analysis \
    --input ${RESULTS}/${search}/*/*-tree-pairs.pkl ${RESULTS}/${search}/*-tree-pairs.pkl \
    --output_dir ${analysis_dir}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}/"
  mkdir -p ${analysis_dir}

  echo "Pipelines in trace analysis: ${search}"
  python -m janus.analysis.pipelines_analysis \
    --input ${RESULTS}/${search}/*-pipelines.pkl > "${analysis_dir}/pipelines_stats.txt"
done
