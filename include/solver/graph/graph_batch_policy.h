#ifndef HPRLP_GRAPH_BATCH_POLICY_H
#define HPRLP_GRAPH_BATCH_POLICY_H

constexpr int HPRLP_NORMAL_GRAPH_BATCH_SIZE = 24;

inline long long hprlp_output_step(long long iter) {
    if (iter <= 0) {
        return 10;
    }
    long long magnitude = 1;
    while (iter >= 10) {
        iter /= 10;
        magnitude *= 10;
    }
    const long long output_step = magnitude / 10;
    return output_step < 10 ? 10 : output_step;
}

inline bool hprlp_iteration_needs_check(long long iter,
                                        int check_interval) {
    if (iter <= 0 || check_interval <= 0) {
        return true;
    }
    return iter % check_interval == 0 ||
           iter % hprlp_output_step(iter) == 0;
}

inline bool hprlp_can_batch_normal_updates(long long current_iter,
                                           int batch_size,
                                           int check_interval) {
    if (current_iter < 0 || batch_size <= 0 || check_interval <= 0) {
        return false;
    }
    for (int offset = 1; offset <= batch_size; ++offset) {
        if (hprlp_iteration_needs_check(current_iter + offset,
                                        check_interval)) {
            return false;
        }
    }
    return true;
}

#endif
