#!/usr/env/bin bash
source scripts/run_setup.sh

# We run kaggle scripts in a docker
# container (with non-root user)
# to provide some basic level of isolation.
# You may prefer to run in a full VM
# If so you can run the `run_kaggle_lift.sh`
# script directly in the VM, and ignore this
# docker wrapper

# configuration based on our large
# rhino server
NUM_CPUS=20
MAX_MEM=150g


mkdir -p "${KAGGLE_FOLDER}/pipelines/"

docker run \
  -it \
  --cpus ${NUM_CPUS} \
  --memory ${MAX_MEM} \
  -v "$(realpath ${KAGGLE_FOLDER})":"/home/${DOCKER_USER}/kaggle/" \
  --entrypoint bash \
  ${DOCKER_CONTAINER_NAME}
