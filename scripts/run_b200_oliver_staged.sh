#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${HPRLP_OLIVER_DATA_DIR:-/home/hprlp/data/Oliver_Hinder}"
RESULT_ROOT="${1:-/home/hprlp/Results/Aug20_HPRLPC_0.1.3_cuda_compat_CUDA133_B200_Oliver_staged_4800s}"
DEVICE="${2:-0}"
JULIA_BIN="${JULIA_BIN:-/home/hprlp/.local/bin/julia}"
START_STAGE="${START_STAGE:-1}"

TIME_LIMIT=4800
TOLERANCE=1e-6

if [[ ! -x "${JULIA_BIN}" ]]; then
    echo "Julia executable not found: ${JULIA_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Oliver_Hinder data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

cd "${REPO_DIR}"

# Intentionally use the public default build command. On the B200 host it
# auto-detects the GPU architecture and the CUDA 13.3 toolkit.
make

STAGING_ROOT="${RESULT_ROOT}/input_groups"

# Run one instance per Julia invocation. Reusing one worker after a very large
# MPS parse can retain reader/model state and make a subsequent large read fail.
# Fresh workers also make every partial result independently resumable.
STAGES=(
    "01_mcf_2500_100_500:mcf_2500_100_500.mps.gz"
    "02_mcf_5000_100_400:mcf_5000_100_400.mps.gz"
    "03_mcf_5000_50_500:mcf_5000_50_500.mps.gz"
    "04_prod_100_300_02:prod_100_300_02.mps.gz"
    "05_heat_source_easy:heat-source-easy.mps.gz"
    "06_mediterranean_shipping:mediterranean-shipping.mps.gz"
    "07_production_inventory:production-inventory.mps.gz"
    "08_qap_tho_150:qap-tho-150.mps.gz"
    "09_qap_wil_100:qap-wil-100.mps.gz"
    "10_supply_chain:supply-chain.mps.gz"
    "11_synthetic_design_match:synthetic-design-match.mps.gz"
    "12_tsp_gaia_10m:tsp-gaia-10m.mps.gz"
)

run_stage() {
    local stage_name="$1"
    local input_dir="$2"
    local result_dir="${RESULT_ROOT}/${stage_name}"

    mkdir -p "${result_dir}"
    echo
    echo "[$(date '+%F %T')] Starting ${stage_name}"
    echo "  input:   ${input_dir}"
    echo "  results: ${result_dir}"

    "${JULIA_BIN}" scripts/run_dataset.jl \
        "${input_dir}" \
        "${result_dir}" \
        --device "${DEVICE}" \
        --time-limit "${TIME_LIMIT}" \
        --tol "${TOLERANCE}" \
        --print-debug-info true \
        --resume
}

mkdir -p "${RESULT_ROOT}"
for stage_spec in "${STAGES[@]}"; do
    stage_name="${stage_spec%%:*}"
    instance_file="${stage_spec#*:}"
    stage_number="${stage_name%%_*}"
    if (( 10#${stage_number} < START_STAGE )); then
        echo "Skipping completed stage ${stage_name}"
        continue
    fi
    source_path="${DATA_DIR}/${instance_file}"
    input_dir="${STAGING_ROOT}/${stage_name}"

    if [[ ! -r "${source_path}" ]]; then
        echo "Missing or unreadable instance: ${source_path}" >&2
        exit 1
    fi
    mkdir -p "${input_dir}"
    ln -sfn "${source_path}" "${input_dir}/${instance_file}"
    run_stage "${stage_name}" "${input_dir}"
done

echo
echo "[$(date '+%F %T')] All Oliver_Hinder stages finished."
echo "Results: ${RESULT_ROOT}"
