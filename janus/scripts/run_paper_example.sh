#!/usr/bin/env bash
source scripts/run_setup.sh
source scripts/utils.sh

#
SEED=42
output_dir=$(realpath "${ANALYSIS}/system-diagram")
mkdir -p ${output_dir}
output_path=${output_dir}/paper_example.output

python -m janus.evaluation.paper_example \
  --input_dir ${RESULTS} \
  --output_dir ${output_dir} \
  --seed ${SEED} \
  | tee ${output_path}


pushd ${ANALYSIS}
# launch local server
php -S localhost:8080 &
webserver_pid=$!

for f in $(ls "system-diagram"/*.html)
do
    png_name="${output_dir}/$(basename ${f} .html).png"
    cropped_png_name="${output_dir}/$(basename ${f} .html)-cropped.png"
    # screenshot of the html
    firefox --screenshot ${png_name} "http://localhost:8080/${f}"
    # crop the image
    convert -trim ${png_name} ${cropped_png_name}
done

popd
kill ${webserver_pid}
