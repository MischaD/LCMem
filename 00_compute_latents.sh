#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Configuration
# --------------------------
PYTHON="python"
PROG="./src/compute_latents.py"
EXP_PATH="./src/base_experiment.py"
EXP_NAME="compute_latents_all"
# Update BASEDIR to point to your data directory
BASEDIR="/path/to/data"
# BASEDIR="/vol/ideadata/ed52egek/pycharm/syneverything/datasets/data"
OUTDIR="./outputs/Latents"
MASTER_PORT=12344
BATCH_SIZE=16

#VALID_DATASETS=("isic_clean" "celeba" "cxr-lt" "mimic" "imagenet_lt" "ctrate_slices_final")
VALID_DATASETS=("ctrate_slices_final")

# --------------------------
# Main loop
# --------------------------
for DATASET in "${VALID_DATASETS[@]}"; do
    # FILELIST="/vol/ideadata/ed52egek/pycharm/syneverything/datasets/${DATASET}.csv"
    FILELIST="./datasets/${DATASET}.csv"
    TARBALL_NAME="${DATASET}_latents_and_stats.tar.gz"
    STATS_NAME="${DATASET}"

    echo "=== Running compute_latents for dataset: ${DATASET} ==="
    echo "Filelist: ${FILELIST}"
    echo "----------------------------------------------"

    CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=$(pwd) \
    "$PYTHON" "$PROG" \
        "$EXP_PATH" \
        "$EXP_NAME" \
        --basedir "$BASEDIR" \
        --output_latents "$OUTDIR" \
        --filelist "$FILELIST" \
        --tarball_name "$TARBALL_NAME" \
        --stats_name "$STATS_NAME" \
        --batch_size "$BATCH_SIZE" \
        --master_port "$MASTER_PORT"

    echo "=== Finished ${DATASET} ==="
    echo
done

echo "All datasets processed."