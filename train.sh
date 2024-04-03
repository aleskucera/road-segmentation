#!/bin/bash

#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks-per-node=1               # 1 task per node
#SBATCH --time=1-00:00:00                 # time limits: 1 day
#SBATCH --error=train_model.out           # standard error file
#SBATCH --output=train_model.out          # standard output file
#SBATCH --partition=amdgpu                # partition name
#SBATCH --cpus-per-task=8                 # number of CPUs
#SBATCH --mail-user=kuceral4@fel.cvut.cz  # where send info about job
#SBATCH --mail-type=BEGIN,FAIL,END        # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL


# Set default values for variables
batch_size="16"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case $1 in
    -b|--batch-size) batch_size="$2"; shift ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

source .env

WANDB_API_KEY=c54dade10e3c04fca21bf96016298e59b1e030ae python main.py \
            train.batch_size="$batch_size"
            train.num_workers=8