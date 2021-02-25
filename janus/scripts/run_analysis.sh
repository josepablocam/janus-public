#!/usr/bin/env bash
source scripts/run_setup.sh
source scripts/utils.sh


SEARCH_STRATEGIES="tpot random"
REPAIR_STRATEGIES="random-mutation random-janus meta-learning janus"
REPAIR_LABELS="random-mutation:Random-Mutation random-janus:Random-Janus meta-learning:Meta-Learning janus:Janus"
SEED=42

if [[ ! -z "${TEST_CONFIG}" ]]
then
  export RESULTS="../results-test/"
  export ANALYSIS="../analysis-test/"
  mkdir -p ${ANALYSIS}
fi


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
      --stat_comparisons "Janus:Meta-Learning" \
      --compute_distance \
      --output_dir ${analysis_dir} \
     | tee ${output_file}


    echo "Producing rules applied analysis: ${experiment_dir}"
    output_file="${analysis_dir}/rules_applied.output"
    python -m janus.analysis.rule_analysis \
      --input ${experiment_dir}/*synthetic-evaluation*-enumerator-statistics \
      --output_dir ${analysis_dir} \
      | tee ${output_file}


    # analysis_dir="${ANALYSIS}/$(basename ${experiment_dir})/user-scripts/"
    # mkdir -p ${analysis_dir}
    # output_file="${analysis_dir}/performance_analysis.output"
    # echo "Collecting ${output_file}"
    # python -m janus.analysis.performance_analysis \
    #   --input ${experiment_dir}/*code-evaluation*.pkl  \
    #   --strategies ${REPAIR_STRATEGIES} \
    #   --labels ${REPAIR_LABELS} \
    #   --compute_distance \
    #   --output_dir ${analysis_dir} \
    #   | tee ${output_file}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}"
  mkdir -p ${analysis_dir}

  echo "Producing rules extracted analysis: ${search}"
  output_file="${analysis_dir}/rule_analysis.output"

  python -m janus.analysis.rule_analysis \
    --input ${RESULTS}/${search}/*-local-rules.pkl \
    --output_dir ${analysis_dir} \
    --rule_sampler weighted \
    --seed ${SEED} \
    | tee ${output_file}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}/tree-sampling/"
  mkdir -p ${analysis_dir}

  echo "Tree sampling analysis: ${search}"
  output_file="${analysis_dir}/tree_pair_analysis.output"

  python -m janus.analysis.tree_pairs_analysis \
    --input ${RESULTS}/${search}/*/*-tree-pairs.pkl ${RESULTS}/${search}/*-tree-pairs.pkl \
    --output_dir ${analysis_dir} \
    | tee ${output_file}
done


for search in ${SEARCH_STRATEGIES}
do
  analysis_dir="${ANALYSIS}/${search}/"
  mkdir -p ${analysis_dir}

  echo "Pipelines in trace analysis: ${search}"
  output_file="${analysis_dir}/pipelines_stats.output"

  python -m janus.analysis.pipelines_analysis \
    --input ${RESULTS}/${search}/*-pipelines.pkl \
    | tee ${output_file}
done



# dump pipelines JSON
python -m janus.analysis.dump_pipelines \
  --input ${RESULTS}/tpot-pipelines-with-tpot-rules/*synthetic-evaluation-janus.pkl \
  --output ${ANALYSIS}/janus-repairs.json

python -m json.tool ${ANALYSIS}/janus-repairs.json ${ANALYSIS}/janus-repairs-formatted.json
