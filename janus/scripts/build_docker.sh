#!/usr/bin/env bash
source scripts/run_setup.sh

cp -r ../datasets datasets/
cp -r ../kaggle kaggle/
git submodule update --init --recursive
docker build -t ${DOCKER_CONTAINER_NAME} \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) .
rm -rf datasets/
rm -rf kaggle/
