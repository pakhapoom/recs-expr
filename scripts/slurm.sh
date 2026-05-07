#!/bin/bash
#SBATCH --job-name=recs-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/recs-model-%j.out
#SBATCH --error=logs/recs-model-%j.err

set -euo pipefail

usage() {
    echo "Usage: sbatch scripts/slurm.sh [--]src/models/model.py [model args...]"
}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi

if [[ "$#" -lt 1 ]]; then
    usage
    exit 2
fi

model_path="${1#--}"
shift

if [[ "${model_path}" != *.py ]]; then
    echo "Model path must point to a .py file: ${model_path}" >&2
    usage
    exit 2
fi

if [[ ! -f "${model_path}" ]]; then
    echo "Model path does not exist: ${model_path}" >&2
    exit 2
fi

experiment="$(basename "${model_path}" .py)"
experiment="${experiment%_recommender}"
recs_path="data/recs/${experiment}.csv"
model_args=("$@")
has_recs_arg=0

for ((i = 0; i < ${#model_args[@]}; i++)); do
    case "${model_args[$i]}" in
        --recs-path|--recs_path)
            if ((i + 1 >= ${#model_args[@]})); then
                echo "${model_args[$i]} requires a value" >&2
                exit 2
            fi
            recs_path="${model_args[$((i + 1))]}"
            has_recs_arg=1
            ;;
        --recs-path=*|--recs_path=*)
            recs_path="${model_args[$i]#*=}"
            has_recs_arg=1
            ;;
    esac
done

if [[ "${has_recs_arg}" -eq 0 ]] && grep -q -- "--recs-path" "${model_path}"; then
    model_args+=(--recs-path "${recs_path}")
fi

mkdir -p data/recs data/metrics

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Model: ${model_path}"
echo "Recommendations: ${recs_path}"

if ((${#model_args[@]})); then
    srun python "${model_path}" "${model_args[@]}"
else
    srun python "${model_path}"
fi
srun python src/evaluate.py --recs_path "${recs_path}"

echo "Finished: $(date)"
