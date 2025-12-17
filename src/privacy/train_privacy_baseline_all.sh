#!/usr/bin/env bash
set -euo pipefail

cd /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/vol/ideadata/ed52egek/pycharm/syneverything/privacy/packhaus \
/vol/ideadata/ed52egek/miniconda/envs/syneverything/bin/python main.py \
  --config_path ./ \
  --config /vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/config_finetune_strongaug_distill.json \
  --filelist /vol/ideadata/ed52egek/pycharm/syneverything/datasets/alldatasets_trainvalbalanced.csv \
  --basedir /vol/ideadata/ed52egek/pycharm/syneverything/datasets/data

