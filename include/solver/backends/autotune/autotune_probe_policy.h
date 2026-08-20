#ifndef HPRLP_AUTOTUNE_PROBE_POLICY_H
#define HPRLP_AUTOTUNE_PROBE_POLICY_H

#include <algorithm>

constexpr int HPRLP_AUTOTUNE_MAX_TIMED_ITERATIONS = 30;
constexpr int HPRLP_AUTOTUNE_MAX_WARMUP_ITERATIONS = 8;

inline int hprlp_autotune_probe_iterations(int max_iterations,
                                           int check_interval) {
    if (max_iterations <= 0 || check_interval <= 0) {
        return 0;
    }
    return std::min(
        std::min(max_iterations, check_interval),
        HPRLP_AUTOTUNE_MAX_TIMED_ITERATIONS);
}

inline int hprlp_autotune_warmup_iterations(int probe_iterations) {
    if (probe_iterations <= 0) {
        return 0;
    }
    return std::min(probe_iterations,
                    HPRLP_AUTOTUNE_MAX_WARMUP_ITERATIONS);
}

#endif
