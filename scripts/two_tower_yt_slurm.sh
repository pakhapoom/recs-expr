#!/bin/bash
#SBATCH --job-name=two-tower-yt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/two-tower-yt-%j.out
#SBATCH --error=logs/two-tower-yt-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"

srun python src/models/two_tower_yt.py
srun python src/evaluate.py --recs_path data/recs/two-tower-yt.csv

echo "Finished: $(date)"
