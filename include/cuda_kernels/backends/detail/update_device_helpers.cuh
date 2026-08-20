#ifndef HPRLP_UPDATE_DEVICE_HELPERS_CUH
#define HPRLP_UPDATE_DEVICE_HELPERS_CUH

#include "api/structs.h"

#include <cstdint>

namespace hprlp {
namespace cuda_kernels {
namespace detail {

__device__ __forceinline__ HPRLP_FLOAT project_x_with_bounds(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) {
        return value;
    }
    if (bound_type == 1) {
        return fmax(value, lower);
    }
    if (bound_type == 2) {
        return fmin(value, upper);
    }
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ HPRLP_FLOAT project_y_delta(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t bound_type) {
    if (bound_type == 0) {
        return 0.0;
    }
    if (bound_type == 1) {
        return fmax(lower - value, 0.0);
    }
    if (bound_type == 2) {
        return fmin(upper - value, 0.0);
    }
    return fmax(lower - value, fmin(upper - value, 0.0));
}

__device__ __forceinline__ HPRLP_FLOAT decode_biased_u16(
    std::uint16_t encoded, int bias, bool has_escape, int escape_value) {
    return has_escape && encoded == 65535
               ? static_cast<HPRLP_FLOAT>(escape_value)
               : static_cast<HPRLP_FLOAT>(encoded) -
                     static_cast<HPRLP_FLOAT>(bias);
}

}  // namespace detail
}  // namespace cuda_kernels
}  // namespace hprlp

#endif
