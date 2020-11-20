#!/usr/bin/env bash
source scripts/run_setup.sh
python -m pytest --capture=no tests/
