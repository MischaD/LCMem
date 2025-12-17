#!/usr/bin/env bash
set -euo pipefail

# Distillation experiment launcher

# Configuration (feel free to edit paths or export overrides before calling)
PYTHON_BIN=${PYTHON_BIN:-/vol/ideadata/ed52egek/miniconda/envs/syneverything/bin/python}
PROGRAM=${PROGRAM:-/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/main.py}
CWD=${CWD:-/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem}
CONFIG=${CONFIG:-/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/config_finetune_strongaug_distill.json}
CONFIG_PATH=${CONFIG_PATH:-./}
FILELIST=${FILELIST:-/vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_trainvalbalanced.csv}
BASEDIR=${BASEDIR:-/vol/ideadata/ed52egek/pycharm/syneverything/datasets/data}

# Environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONPATH=${PYTHONPATH:-/vol/ideadata/ed52egek/pycharm/syneverything/privacy/packhaus}

echo "Running distillation experiment"
echo "Using device(s): $CUDA_VISIBLE_DEVICES"
echo "Program: $PROGRAM"
echo "Config:  $CONFIG"
echo "Filelist: $FILELIST"
echo "Basedir: $BASEDIR"

cd "$CWD"

"$PYTHON_BIN" "$PROGRAM" \
  --config_path "$CONFIG_PATH" \
  --config "$CONFIG" \
  --filelist "$FILELIST" \
  --basedir "$BASEDIR"


