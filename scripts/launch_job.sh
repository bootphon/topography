#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=job_%A_%j.out
#SBATCH --error=job_%A_%j.err
#SBATCH --time=03:00:00

set -e # fail fully on first line failure
set -x

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# source /shared/apps/anaconda3/etc/profile.d/conda.sh
# conda activate topo

module load python/3.10.4
conda activate topography

echo $JOB_CMD
srun $JOB_CMD
