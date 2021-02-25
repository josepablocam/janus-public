#!/usr/bin/env bash
source scripts/run_setup.sh

cp -r ../datasets datasets/
cp -r ../kaggle kaggle/
git submodule update --init --recursive
docker build -t ${DOCKER_CONTAINER_NAME} .
rm -rf datasets/
rm -rf kaggle/
