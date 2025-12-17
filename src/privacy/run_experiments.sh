#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./run_experiments.sh <model|all> <device>
#
# Datasets (keys): isic, chestxray8, ctrate, imagenetlt, cxrlt, celeba
# Models (keys): baseline, nxt_512, dinov3
# Device: GPU device number (e.g. 0)
# If you pass "all all" (or no args), it will run all combinations (18 runs).

cd /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem

PYTHON="/vol/ideadata/ed52egek/miniconda/envs/syneverything/bin/python"
MAIN_PY="main.py"
CONFIG_DIR="./"
basedir="/vol/ideadata/ed52egek/pycharm/syneverything/datasets/data"
filelist="/vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_trainvalbalanced.csv"

declare -A MODEL_CONFIG
MODEL_CONFIG[camera_ready_packhaus]="camera_ready_packhaus.json"
MODEL_CONFIG[camera_ready_dar]="camera_ready_dar.json"
MODEL_CONFIG[camera_ready_dar_unsuper]="camera_ready_dar_unsuper.json"

run_one() {
  local model_key="$1"     # e.g., baseline
  local config_file="${MODEL_CONFIG[$model_key]:-}"

  if [[ -z "$basedir" || -z "$filelist" || -z "$config_file" ]]; then
    echo "[ERROR] Unknown dataset '${dataset_key}' or model '${model_key}'." >&1
    return 1
  fi

  if [[ ! -f "$filelist" ]]; then
    echo "[WARN] FILELIST not found: $filelist" >&1
  fi

  if [[ "$model_key" == "camera_ready_dar_unsuper" ]]; then
    UNSUPERVISED="--unsupervised"
  else
    UNSUPERVISED=""
  fi

  local exp_desc="${model_key}"
  export CUDA_VISIBLE_DEVICES="$DEVICE";
  echo "[RUN] model=${model_key} config=${config_file}" >&1
   "$PYTHON" "$MAIN_PY" \
    --config_path "$CONFIG_DIR/" \
    --config "$config_file" \
    --filelist "$filelist" \
    --basedir "$basedir" \
    --resume_checkpoint "/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/${model_key}/${model_key}_checkpoint.pth" \
    --experiment_description "$exp_desc" \
    $UNSUPERVISED
}

MODEL_INPUT="${1:-all}"
DEVICE="${2:-0}"


run_one "$MODEL_INPUT"
