namespace {

BackendCandidate choose_backend(HPRLP_FLOAT ref_metric,
                                long long ref_time_ns,
                                const std::vector<std::pair<BackendCandidate, std::pair<HPRLP_FLOAT, long long>>> &candidates) {
    BackendCandidate choice{HPRLPXBackend::ScaledCusparse,
                            HPRLPYBackend::ScaledCusparse};
    HPRLP_FLOAT allowed_metric = ref_metric +
        std::max(static_cast<HPRLP_FLOAT>(1e-12),
                 std::abs(ref_metric) * static_cast<HPRLP_FLOAT>(1e-8));
    double best_time = std::numeric_limits<double>::infinity();
    for (const auto &candidate : candidates) {
        HPRLP_FLOAT metric = candidate.second.first;
        long long time_ns = candidate.second.second;
        if (!std::isfinite(metric) || metric > allowed_metric || time_ns <= 0) {
            continue;
        }
        if (time_ns <= static_cast<long long>(ref_time_ns * 0.95) && static_cast<double>(time_ns) < best_time) {
            best_time = static_cast<double>(time_ns);
            choice = candidate.first;
        }
    }
    return choice;
}

bool same_backend_candidate(const BackendCandidate &lhs,
                            const BackendCandidate &rhs) {
    return lhs.x_backend == rhs.x_backend &&
           lhs.y_backend == rhs.y_backend;
}

bool uses_fixed_degree_backend(const BackendCandidate &candidate) {
    return candidate.x_backend ==
               HPRLPXBackend::FixedDegreePackedDictionary ||
           candidate.y_backend ==
               HPRLPYBackend::FixedDegreePackedDictionary;
}

BackendCandidate choose_backend_with_fixed_degree_gate(
    const std::vector<std::pair<
        BackendCandidate, std::pair<HPRLP_FLOAT, long long>>> &results) {
    const HPRLP_FLOAT ref_metric = results.front().second.first;
    const long long ref_time_ns = results.front().second.second;
    std::vector<std::pair<
        BackendCandidate, std::pair<HPRLP_FLOAT, long long>>>
        incumbent_candidates;
    std::vector<std::pair<
        BackendCandidate, std::pair<HPRLP_FLOAT, long long>>>
        fixed_degree_candidates;
    for (std::size_t index = 1; index < results.size(); ++index) {
        if (uses_fixed_degree_backend(results[index].first)) {
            fixed_degree_candidates.push_back(results[index]);
        } else {
            incumbent_candidates.push_back(results[index]);
        }
    }

    const BackendCandidate incumbent = choose_backend(
        ref_metric, ref_time_ns, incumbent_candidates);
    long long incumbent_time_ns = ref_time_ns;
    for (const auto &result : results) {
        if (same_backend_candidate(result.first, incumbent)) {
            incumbent_time_ns = result.second.second;
            break;
        }
    }

    const BackendCandidate fixed_degree_choice = choose_backend(
        ref_metric, ref_time_ns, fixed_degree_candidates);
    if (!uses_fixed_degree_backend(fixed_degree_choice)) {
        return incumbent;
    }
    long long fixed_degree_time_ns =
        std::numeric_limits<long long>::max();
    for (const auto &result : results) {
        if (same_backend_candidate(result.first, fixed_degree_choice)) {
            fixed_degree_time_ns = result.second.second;
            break;
        }
    }
    return fixed_degree_time_ns <=
               static_cast<long long>(incumbent_time_ns * 0.95)
        ? fixed_degree_choice
        : incumbent;
}

} // namespace
