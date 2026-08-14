#!/bin/bash
# run_cuopt.sh
#
# Runs the cuOpt GPU solver on every instance in testcase/ at four time
# limits (60s, 10s, 5s, 2s) and writes results to outputs_cuopt/cuopt_results.csv
#
# Usage:
#   bash run_cuopt.sh

set -euo pipefail

TESTCASE_DIR="testcase"
OUTPUT_DIR="outputs_cuopt"
OUTPUT_CSV="${OUTPUT_DIR}/cuopt_results.csv"
TIMEOUTS=(60 10 5 2)

mkdir -p "${OUTPUT_DIR}"

echo "Running cuOpt over instances in ${TESTCASE_DIR}/ with timeouts: ${TIMEOUTS[*]} seconds"

python3 solve_cuopt.py \
    --testcase_dir "${TESTCASE_DIR}" \
    --output_csv "${OUTPUT_CSV}" \
    --timeouts "${TIMEOUTS[@]}"

echo "All instances processed. Results are in ${OUTPUT_CSV}"
