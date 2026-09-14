#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
detector="$repo_root/scripts/detect_cuda_arch.sh"

expect_arch() {
    local expected=$1
    shift
    local actual
    actual=$(bash "$detector" "$@")
    if [[ $actual != "$expected" ]]; then
        printf 'expected sm_%s, got sm_%s for: %s\n' \
            "$expected" "$actual" "$*" >&2
        exit 1
    fi
}

expect_arch 100 --from-compute-cap 10.0
expect_arch 90 --from-compute-cap 9.0
expect_arch 80 --from-compute-cap 8.0
expect_arch 86 --from-compute-cap 8.6
expect_arch 89 --from-compute-cap 8.9
expect_arch 120 --from-compute-cap 12.0

expect_arch 100 --from-name "NVIDIA B200"
expect_arch 90 --from-name "NVIDIA H100 80GB HBM3"
expect_arch 80 --from-name "NVIDIA A100-SXM4-80GB"
expect_arch 86 --from-name "NVIDIA GeForce RTX 3090"
expect_arch 89 --from-name "NVIDIA GeForce RTX 4090"
expect_arch 120 --from-name "NVIDIA GeForce RTX 5090"

fake_dir=$(mktemp -d)
trap 'rm -rf "$fake_dir"' EXIT
fake_smi="$fake_dir/nvidia-smi"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ $* == *compute_cap* ]]; then' \
    '    [[ -n ${FAKE_COMPUTE_CAP:-} ]] || exit 1' \
    '    printf "%s\\n" "$FAKE_COMPUTE_CAP"' \
    'elif [[ $* == *name* ]]; then' \
    '    printf "%s\\n" "${FAKE_GPU_NAME:-}"' \
    'fi' >"$fake_smi"
chmod +x "$fake_smi"

actual=$(FAKE_COMPUTE_CAP=8.9 NVIDIA_SMI="$fake_smi" bash "$detector")
[[ $actual == 89 ]] || {
    printf 'nvidia-smi compute-capability detection returned sm_%s\n' "$actual" >&2
    exit 1
}

actual=$(FAKE_GPU_NAME="NVIDIA GeForce RTX 5090" \
    NVIDIA_SMI="$fake_smi" bash "$detector")
[[ $actual == 120 ]] || {
    printf 'nvidia-smi name fallback returned sm_%s\n' "$actual" >&2
    exit 1
}

printf 'detect_cuda_arch_test: PASS\n'
