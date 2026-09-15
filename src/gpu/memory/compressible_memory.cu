#include "gpu/memory/compressible_memory.h"
#include "solver/constants.h"

#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {

struct CompressibleAllocation {
    std::size_t bytes = 0;
    CUmemGenericAllocationHandle handle = 0;
    CUcontext context = nullptr;
    bool mapped = false;
    bool handle_live = false;
    bool address_reserved = false;
};

std::mutex allocation_mutex;
std::unordered_map<const void *, CompressibleAllocation> compressible_allocations;
thread_local HPRLP_compressible_memory_mode compressible_memory_mode =
    HPRLP_COMPRESSIBLE_MEMORY_FROM_ENVIRONMENT;
constexpr std::size_t kMinCompressibleBytes = 1024 * 1024;

cudaError_t fallback_malloc(void **ptr, std::size_t bytes) {
    return cudaMalloc(ptr, bytes);
}

cudaError_t driver_failure(CUresult status) {
    return status == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

bool current_driver_state(CUdevice *device, CUcontext *context) {
    if (cudaFree(nullptr) != cudaSuccess ||
        cuCtxGetDevice(device) != CUDA_SUCCESS ||
        cuCtxGetCurrent(context) != CUDA_SUCCESS) {
        return false;
    }
    return *context != nullptr;
}

bool device_supports_compression(CUdevice device) {
#if CUDA_VERSION >= 11020
    int vmm_supported = 0;
    int compression_supported = 0;
    return cuDeviceGetAttribute(&vmm_supported,
                                CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                device) == CUDA_SUCCESS &&
           vmm_supported != 0 &&
           cuDeviceGetAttribute(&compression_supported,
                                CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
                                device) == CUDA_SUCCESS &&
           compression_supported != 0;
#else
    (void)device;
    return false;
#endif
}

CUresult cleanup_vmm(CUdeviceptr address, CompressibleAllocation *allocation) {
    if (allocation->mapped) {
        const CUresult status = cuMemUnmap(address, allocation->bytes);
        if (status != CUDA_SUCCESS) return status;
        allocation->mapped = false;
    }
    if (allocation->handle_live) {
        const CUresult status = cuMemRelease(allocation->handle);
        if (status != CUDA_SUCCESS) return status;
        allocation->handle_live = false;
    }
    if (allocation->address_reserved) {
        const CUresult status = cuMemAddressFree(address, allocation->bytes);
        if (status != CUDA_SUCCESS) return status;
        allocation->address_reserved = false;
    }
    return CUDA_SUCCESS;
}

cudaError_t rollback_or_fallback(void **ptr,
                                 std::size_t requested_bytes,
                                 CUdeviceptr address,
                                 CompressibleAllocation *allocation) {
    const CUresult rollback_status = cleanup_vmm(address, allocation);
    if (rollback_status != CUDA_SUCCESS) {
        return driver_failure(rollback_status);
    }
    return fallback_malloc(ptr, requested_bytes);
}

}  // namespace

bool hprlp_device_supports_compression() {
    CUdevice device;
    CUcontext context;
    return current_driver_state(&device, &context) && device_supports_compression(device);
}

HPRLP_compressible_memory_mode hprlp_get_compressible_memory_mode() {
    return compressible_memory_mode;
}

void hprlp_set_compressible_memory_mode(
        HPRLP_compressible_memory_mode mode) {
    compressible_memory_mode = mode;
}

bool hprlp_compressible_memory_requested() {
    if (compressible_memory_mode == HPRLP_COMPRESSIBLE_MEMORY_ENABLED) {
        return true;
    }
    if (compressible_memory_mode == HPRLP_COMPRESSIBLE_MEMORY_DISABLED) {
        return false;
    }
    const char *disable_compression =
        std::getenv("HPRLP_DISABLE_COMPRESSIBLE_MEMORY");
    if (disable_compression != nullptr &&
        std::strcmp(disable_compression, "0") != 0) {
        return false;
    }
    const char *enable_compression =
        std::getenv("HPRLP_ENABLE_COMPRESSIBLE_MEMORY");
    if (enable_compression == nullptr) {
        return hprlp::constants::DEFAULT_ENABLE_COMPRESSIBLE_MEMORY;
    }
    return std::strcmp(enable_compression, "0") != 0;
}

cudaError_t hprlp_device_malloc_compressible(void **ptr, std::size_t bytes) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;
    if (bytes == 0) {
        return cudaSuccess;
    }
    if (bytes < kMinCompressibleBytes) {
        return fallback_malloc(ptr, bytes);
    }
    if (!hprlp_compressible_memory_requested()) {
        return fallback_malloc(ptr, bytes);
    }

    CUdevice device;
    CUcontext context;
    if (!current_driver_state(&device, &context) || !device_supports_compression(device)) {
        return fallback_malloc(ptr, bytes);
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    std::size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
        granularity == 0 || bytes > std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
        return fallback_malloc(ptr, bytes);
    }
    const std::size_t rounded_bytes = ((bytes + granularity - 1) / granularity) * granularity;

    CUdeviceptr address = 0;
    CompressibleAllocation allocation;
    allocation.bytes = rounded_bytes;
    allocation.context = context;

    if (cuMemAddressReserve(&address, rounded_bytes, 0, 0, 0) != CUDA_SUCCESS) {
        return fallback_malloc(ptr, bytes);
    }
    allocation.address_reserved = true;

    if (cuMemCreate(&allocation.handle, rounded_bytes, &prop, 0) != CUDA_SUCCESS) {
        return rollback_or_fallback(ptr, bytes, address, &allocation);
    }
    allocation.handle_live = true;

    CUmemAllocationProp actual_prop = {};
    if (cuMemGetAllocationPropertiesFromHandle(&actual_prop, allocation.handle) != CUDA_SUCCESS ||
        actual_prop.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
        return rollback_or_fallback(ptr, bytes, address, &allocation);
    }

    if (cuMemMap(address, rounded_bytes, 0, allocation.handle, 0) != CUDA_SUCCESS) {
        return rollback_or_fallback(ptr, bytes, address, &allocation);
    }
    allocation.mapped = true;

    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(address, rounded_bytes, &access, 1) != CUDA_SUCCESS) {
        return rollback_or_fallback(ptr, bytes, address, &allocation);
    }

    try {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        const auto inserted = compressible_allocations.emplace(
            reinterpret_cast<const void *>(address), allocation);
        if (!inserted.second) {
            return rollback_or_fallback(ptr, bytes, address, &allocation);
        }
    } catch (...) {
        const CUresult rollback_status = cleanup_vmm(address, &allocation);
        return rollback_status == CUDA_SUCCESS ? cudaErrorMemoryAllocation : driver_failure(rollback_status);
    }

    *ptr = reinterpret_cast<void *>(address);
    return cudaSuccess;
}

cudaError_t hprlp_device_free(void *ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }

    std::unique_lock<std::mutex> lock(allocation_mutex);
    auto entry = compressible_allocations.find(ptr);
    if (entry == compressible_allocations.end()) {
        lock.unlock();
        return cudaFree(ptr);
    }
    const CUcontext allocation_context = entry->second.context;

    CUcontext current_context = nullptr;
    CUresult context_status = cuCtxGetCurrent(&current_context);
    if (context_status != CUDA_SUCCESS) {
        return driver_failure(context_status);
    }

    bool pushed_context = false;
    if (current_context != allocation_context) {
        context_status = cuCtxPushCurrent(allocation_context);
        if (context_status != CUDA_SUCCESS) {
            return driver_failure(context_status);
        }
        pushed_context = true;
    }

    const CUresult cleanup_status = cleanup_vmm(reinterpret_cast<CUdeviceptr>(ptr), &entry->second);
    if (cleanup_status == CUDA_SUCCESS) {
        compressible_allocations.erase(entry);
    }

    CUresult pop_status = CUDA_SUCCESS;
    if (pushed_context) {
        CUcontext popped_context = nullptr;
        pop_status = cuCtxPopCurrent(&popped_context);
        if (pop_status == CUDA_SUCCESS && popped_context != allocation_context) {
            pop_status = CUDA_ERROR_INVALID_CONTEXT;
        }
    }
    lock.unlock();

    if (cleanup_status != CUDA_SUCCESS) return driver_failure(cleanup_status);
    return driver_failure(pop_status);
}

bool hprlp_is_compressible_allocation(const void *ptr) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    return compressible_allocations.find(ptr) != compressible_allocations.end();
}

std::size_t hprlp_compressible_allocation_count() {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    return compressible_allocations.size();
}
