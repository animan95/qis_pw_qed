#!/bin/bash
#SBATCH --job-name=qeom44
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=price
#SBATCH --clusters=amareln
#SBATCH --exclude=gpun[001-003]
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/qeom.%j.out
#SBATCH --error=logs/qeom.%j.err
set -euo pipefail
export XDG_RUNTIME_DIR=/scratch/am4655/tmp; mkdir -p "$XDG_RUNTIME_DIR"
source /projectsn/mp1009_1/am4655/qis_pw_qed/env_qis.sh
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export QIS_RUN_QEOM=1 QIS_NCAS=4 QIS_NELECAS=4
cd /projectsn/mp1009_1/am4655/qis_pw_qed
echo "[qeom] start $(date) on $(hostname)"
python -u scripts/phase0_licn_gas.py &> logs/phase0_licn_gas_qeom_cas44.log
echo "[qeom] done $(date)"
