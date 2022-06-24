#!/bin/bash

jobs_in_parallel=128

if [ ! -f "$1" ]
then
    echo "Error: file passed does not exist"
    exit 1
fi

# This convoluted way of counting also works if a final EOL character is missing
n_lines=$(grep -c '^' "$1")

# Use file name for job name
job_name=$(basename "$1" .txt)

sbatch --array=1-${n_lines}%${jobs_in_parallel} \
--job-name ${job_name} $(dirname "$0")/launch_job.sh \
--output=$WORK/logs/%x_%j.out \
--error=$WORK/logs/%x_%j.err \
"$1"