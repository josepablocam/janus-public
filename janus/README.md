# Janus

Janus is the Roman god of time (or more importantly, transitions/change).
`janus` is
a tool that learns edits to *change* an input ML pipeline to improve it
(i.e. repair it).

# Docker
We recommend running this on your bare machine (i.e. not through docker). We've been running this on Ubuntu 16.04 server (40 cores, 500GB RAM).
However, you may want to use the docker image for a quick test. If so,
we assume you have docker installed and you can run:

```
git submodule update
bash scripts/build_docker.sh
```

You can then launch the container using

```
docker run -it --rm janus-container
# inside the container
# confirm build worked
$ bash scripts/run_tests.sh
```

You can skip all install steps described below, as the
docker build has taken care of them.

# Install
If you are installing on your machine (rather than using
the docker container), please run:

```
bash scripts/install.sh
```

will download/install conda if not available and setup the
conda environment.

You can confirm the project is working as expected by running
the associated tests

```
bash scripts/run_tests.sh
```

# Reproduce

You can run

```
bash scripts/reproduce.sh
```

to run data collection and experiments. If you'd like to run
one at a time, we provide details below.

## Data Collection
`janus` collects pipelines produced by an AutoML system (TPOT),
pairs trees that are "nearby" (in tree edit distance) and where
one of the trees outperforms the other, extracts tree edits,
and converts these into local edit rules.

```
bash scripts/run_collect_data.sh
```

## Experiments
Our basic evaluation consists of: comparing different rule derivation/applications
to random mutations (which is what existing GA AutoML tools use).

```
bash scripts/run_experiments.sh
```

## Quick Testing
If you'd like to run scripts with reduced time (i.e. just to make sure nothing is broken), you can run define the environment variable `TEST_CONFIG=1`. For example,

```
TEST_CONFIG=1 bash scripts/reproduce.sh
```

will run the entire pipeline with a test configuration.

## Project Tests
You can run project tests with

```
bash scripts/run_tests.sh
```
