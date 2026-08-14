#!/bin/bash
#SBATCH --job-name=cuopt_cvrptw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=cuopt_job.%J.out
#SBATCH --error=cuopt_job.%J.err

cd "$SLURM_SUBMIT_DIR"

# --- One-time setup (run these on the LOGIN node, not inside this job) ---
#   python3 -m venv ~/cuopt_env
#   source ~/cuopt_env/bin/activate
#   pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'
# (use cuopt-cu13 instead if the cluster's CUDA runtime is 13.x - check with
#  `nvcc --version` / `nvidia-smi` on a GPU node first)
# ---------------------------------------------------------------------------
source /home/apps/spack/share/spack/setup-env.sh
spack load python@3.12.13
source ~/cuopt_env/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv

bash run_cuopt.sh
