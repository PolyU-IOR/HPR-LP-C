#ifndef HPRLP_COMPRESSIBLE_MEMORY_H
#define HPRLP_COMPRESSIBLE_MEMORY_H

#include <cuda_runtime.h>
#include <cstddef>

enum HPRLP_compressible_memory_mode {
    HPRLP_COMPRESSIBLE_MEMORY_FROM_ENVIRONMENT = -1,
    HPRLP_COMPRESSIBLE_MEMORY_DISABLED = 0,
    HPRLP_COMPRESSIBLE_MEMORY_ENABLED = 1,
};

bool hprlp_device_supports_compression();
HPRLP_compressible_memory_mode hprlp_get_compressible_memory_mode();
void hprlp_set_compressible_memory_mode(HPRLP_compressible_memory_mode mode);
bool hprlp_compressible_memory_requested();
cudaError_t hprlp_device_malloc_compressible(void **ptr, std::size_t bytes);
cudaError_t hprlp_device_free(void *ptr);
bool hprlp_is_compressible_allocation(const void *ptr);
std::size_t hprlp_compressible_allocation_count();

template <typename T>
inline cudaError_t hprlp_device_malloc_compressible(T **ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    void *allocation = nullptr;
    const cudaError_t status = hprlp_device_malloc_compressible(&allocation, bytes);
    if (status == cudaSuccess) {
        *ptr = static_cast<T *>(allocation);
    }
    return status;
}

#endif
