#!/bin/bash
#SBATCH --job-name=convnext_array
#SBATCH --gpus=1
#SBATCH --time=23:59:00         # Hours:Mins:Secs

# -------------------------------------------------------------------------------------------------------------------------------- 
# ----------------------------------------------------- Hyperparameters ---------------------------------------------------------- 
# -------------------------------------------------------------------------------------------------------------------------------- 

# Define ConvNeXt variants array
CONVEXT_VARIANTS=("ConvNeXt-Tiny" "ConvNeXt-Small" "ConvNeXt-Base" "ConvNeXt-Large")
CONFIG_FILES=("clm_convnext_tiny.json" "clm_convnext_small.json" "clm_convnext_base.json" "clm_convnext_large.json")

# Get the current variant based on SLURM_ARRAY_TASK_ID (if running as array job)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CURRENT_VARIANT=${CONVEXT_VARIANTS[$SLURM_ARRAY_TASK_ID]}
    CURRENT_CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
    EXPERIMENT_DESCRIPTION="clm_convnext_${CURRENT_VARIANT,,}"  # Convert to lowercase
    echo "Array Job - Task ID: $SLURM_ARRAY_TASK_ID, Variant: $CURRENT_VARIANT"
else
    # Default to ConvNeXt-Tiny if not running as array job
    CURRENT_VARIANT="ConvNeXt-Tiny"
    CURRENT_CONFIG="clm_convnext_tiny.json"
    EXPERIMENT_DESCRIPTION="clm_convnext_convnext-tiny"
    echo "Single Job - Running ConvNeXt-Tiny"
fi

RESUME_CHECKPOINT=""

# activate conda 
# -------------------------------------------------------------------------------------------------------------------------------- 
# source ~/miniforge3/bin/activate
# conda activate syneverything
# -------------------------------------------------------------------------------------------------------------------------------- 


# unpacking datasets and preparing datasets
# -------------------------------------------------------------------------------------------------------------------------------- 
# DATA_BASE_DIR=/projects/u5cy/syneverything
DATA_BASE_DIR=./data
LATENT_BASE_DIR=$DATA_BASE_DIR/latents/


# -------------------------------------------------------------------------------------------------------------------------------- 

cd ./src/privacy/
export PYTHONPATH=$PWD
export N_GPUS=${SLURM_GPUS_ON_NODE:-1}  # Default to 1 if not set
if [ "$N_GPUS" -gt 1 ]; then
    DISTRIBUTED="--distributed"
else
    DISTRIBUTED=""
fi
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    RESUME_CHECKPOINT="--resume_checkpoint $RESUME_CHECKPOINT"
else
    RESUME_CHECKPOINT=""
fi

# Define the parameters for each run
# Define the parameters for each run
BASE_DIR="."
PROGRAM="./main.py"
FILELIST="$DATA_BASE_DIR/alldatasets_trainvalbalanced.csv";

# Export PYTHONPATH
export PYTHONPATH=$BASE_DIR

CMD="$PROGRAM \
    --config_path $BASE_DIR \
    --config $CURRENT_CONFIG \
    --filelist $FILELIST \
    --basedir $LATENT_BASE_DIR \
    --experiment_description $EXPERIMENT_DESCRIPTION \
    $DISTRIBUTED \
    $RESUME_CHECKPOINT"


echo "Executing command: python ${CMD}"
python $CMD