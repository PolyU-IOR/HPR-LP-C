#ifndef HPRLP_UNIT_OPERATOR_POLICY_H
#define HPRLP_UNIT_OPERATOR_POLICY_H

#include <cstddef>
#include <cmath>

inline bool hprlp_use_unit_operator_x(int uniform_unit_sign,
                                      std::size_t constraint_count,
                                      std::size_t nonzeros) {
    return uniform_unit_sign != 0 && nonzeros > 0 &&
           constraint_count <= 65536;
}

inline bool hprlp_use_unit_operator_y(int uniform_unit_sign,
                                      std::size_t nonzeros) {
    return uniform_unit_sign != 0 && nonzeros > 0;
}

template <typename T>
inline int hprlp_uniform_unit_sign(const T *values,
                                   std::size_t value_count) {
    if (values == nullptr || value_count == 0 ||
        (values[0] != T(1) && values[0] != T(-1))) {
        return 0;
    }
    const T sign = values[0];
    for (std::size_t index = 1; index < value_count; ++index) {
        if (!std::isfinite(static_cast<double>(values[index])) ||
            values[index] != sign) {
            return 0;
        }
    }
    return sign == T(1) ? 1 : -1;
}

template <typename T>
inline bool hprlp_all_zero_lower_unbounded(const T *lower,
                                           const T *upper,
                                           std::size_t variable_count) {
    if (lower == nullptr || upper == nullptr || variable_count == 0) {
        return false;
    }
    for (std::size_t i = 0; i < variable_count; ++i) {
        if (lower[i] != T(0) || !std::isinf(upper[i]) || upper[i] <= T(0)) {
            return false;
        }
    }
    return true;
}

#endif
