#!/usr/bin/env bash

set -euo pipefail

sm_from_compute_capability() {
    local capability=${1//[[:space:]]/}
    if [[ $capability =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        printf '%s%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi
    return 1
}

sm_from_gpu_name() {
    local gpu_name=${1^^}
    case "$gpu_name" in
        *GB200*|*B200*)
            printf '100\n'
            ;;
        *GH200*|*H200*|*H100*)
            printf '90\n'
            ;;
        *A100*)
            printf '80\n'
            ;;
        *"GEFORCE RTX 50"*)
            printf '120\n'
            ;;
        *"GEFORCE RTX 40"*)
            printf '89\n'
            ;;
        *"GEFORCE RTX 30"*)
            printf '86\n'
            ;;
        *)
            return 1
            ;;
    esac
}

case ${1:-} in
    --from-compute-cap)
        [[ $# -eq 2 ]] || exit 2
        sm_from_compute_capability "$2"
        exit
        ;;
    --from-name)
        [[ $# -eq 2 ]] || exit 2
        sm_from_gpu_name "$2"
        exit
        ;;
    --help)
        printf 'Usage: %s [--from-compute-cap M.m | --from-name GPU-NAME]\n' "$0"
        exit
        ;;
    '')
        ;;
    *)
        exit 2
        ;;
esac

nvidia_smi=${NVIDIA_SMI:-}
if [[ -z $nvidia_smi ]]; then
    nvidia_smi=$(command -v nvidia-smi 2>/dev/null || true)
fi
[[ -n $nvidia_smi && -x $nvidia_smi ]] || exit 0

compute_capability=$(
    "$nvidia_smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
        tr -d ' ' |
        grep -E '^[0-9]+\.[0-9]+$' |
        head -n1 || true
)
if [[ -n $compute_capability ]]; then
    sm_from_compute_capability "$compute_capability"
    exit
fi

gpu_name=$(
    "$nvidia_smi" --query-gpu=name --format=csv,noheader 2>/dev/null |
        head -n1 || true
)
[[ -n $gpu_name ]] || exit 0
sm_from_gpu_name "$gpu_name" || true
