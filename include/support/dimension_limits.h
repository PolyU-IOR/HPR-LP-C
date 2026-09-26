#ifndef HPRLP_DIMENSION_LIMITS_H
#define HPRLP_DIMENSION_LIMITS_H

#include <cstdint>
#include <iostream>
#include <limits>

// The solver and dense kernels still index rows and columns with int32.
// Check original 64-bit dimensions before any narrowing conversion.
inline bool hprlp_dimensions_fit_int32(std::int64_t rows,
                                      std::int64_t columns) {
    if (rows > std::numeric_limits<int>::max() ||
        columns > std::numeric_limits<int>::max()) {
        std::cerr << "[warning] HPR-LP-C does not support m or n above "
                     "INT32_MAX (m=" << rows << ", n=" << columns
                  << "); model not created.\n";
        return false;
    }
    return true;
}

#endif
