#!/usr/bin/env bash
source scripts/run_setup.sh

if [[ ${1} == --download ]]
then
    echo "Downloading new set of kaggle scripts -- otherwise uses"
    echo "those commited to the janus repo"
    bash scripts/run_kaggle_download.sh
    bash scripts/run_kaggle_requirements.sh
fi

bash scripts/build_docker.sh

echo "When docker container has launched"
echo "Please run the following manually"
echo "bash scripts/run_kaggle_lift.sh"

bash scripts/run_docker.sh
