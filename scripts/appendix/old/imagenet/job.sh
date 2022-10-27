#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=90:00:00
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread
#SBATCH --output=imagenet_%j.out
#SBATCH --error=imagenet_%j.err


set -x

module load python/3.10.4
conda activate topography

srun accelerate launch $WORK/topography/scripts/imagenet/run.py --log $SCRATCH/imagenet --data $SCRATCH/data
