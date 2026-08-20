namespace {

constexpr int kFusedThreads = 512;
constexpr int kWarpsPerBlock = kFusedThreads / 32;
constexpr int kSignedScalarThreads = 256;
constexpr int kUnitColTileThreads = 256;
constexpr double kSignedZeroSkipEnableFraction = 0.50;
constexpr double kSignedZeroSkipDisableFraction = 0.35;

__global__ void mark_raw_positive_zero_flags_kernel(
    const HPRLP_FLOAT *values, std::uint8_t *nonzero, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        nonzero[index] = __double_as_longlong(values[index]) == 0ll ? 0 : 1;
    }
}

__global__ void count_raw_positive_zero_flags_kernel(
    const std::uint8_t *nonzero, int count,
    unsigned long long *positive_zero_count) {
    __shared__ unsigned int warp_counts[32];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned int zero_mask = __ballot_sync(
        0xffffffffu, index < count && nonzero[index] == 0);
    if (lane == 0) {
        warp_counts[warp] = __popc(zero_mask);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int block_count = 0;
        const int warp_count = (blockDim.x + 31) >> 5;
        for (int i = 0; i < warp_count; ++i) {
            block_count += warp_counts[i];
        }
        atomicAdd(positive_zero_count,
                  static_cast<unsigned long long>(block_count));
    }
}

struct BackendCandidate {
    HPRLPXBackend x_backend;
    HPRLPYBackend y_backend;
};

const char *x_backend_name(HPRLPXBackend backend) {
    switch (backend) {
        case HPRLPXBackend::ScaledCusparse: return "cusparse";
        case HPRLPXBackend::GenericFused: return "fused";
        case HPRLPXBackend::UnitFactorized: return "unit-factor";
        case HPRLPXBackend::SignedUnitPacked: return "signed-packed";
        case HPRLPXBackend::SignedUnitSplitU16:
            return "signed-split-u16";
        case HPRLPXBackend::PackedDictionary: return "packed-dictionary";
        case HPRLPXBackend::FixedDegreePackedDictionary:
            return "fixed-degree-packed-dictionary";
        case HPRLPXBackend::StructuredOriginal: return "structured";
        case HPRLPXBackend::FactorizedStencil: return "factorized-stencil";
        case HPRLPXBackend::GridSlackLaplacian:
            return "grid-slack-laplacian";
        case HPRLPXBackend::RowTemplate: return "row-template";
        case HPRLPXBackend::AffineBlock: return "affine-block";
    }
    return "unknown";
}

const char *y_backend_name(HPRLPYBackend backend) {
    switch (backend) {
        case HPRLPYBackend::ScaledCusparse: return "cusparse";
        case HPRLPYBackend::GenericFused: return "fused";
        case HPRLPYBackend::SegmentedFused: return "segmented-fused";
        case HPRLPYBackend::UnitFactorized: return "unit-factor";
        case HPRLPYBackend::SignedUnitPacked: return "signed-packed";
        case HPRLPYBackend::SignedUnitPackedCombined:
            return "signed-packed-combined";
        case HPRLPYBackend::StructuredOriginal: return "structured";
        case HPRLPYBackend::FactorizedStencil: return "factorized-stencil";
        case HPRLPYBackend::UnitColTile: return "unit-coltile";
        case HPRLPYBackend::UnitColTileZeroBitset:
            return "unit-coltile-zero-bitset";
        case HPRLPYBackend::UnitActiveScatter:
            return "unit-active-scatter";
        case HPRLPYBackend::GridSlackLaplacian:
            return "grid-slack-laplacian";
        case HPRLPYBackend::PackedDictionary: return "packed-dictionary";
        case HPRLPYBackend::FixedDegreePackedDictionary:
            return "fixed-degree-packed-dictionary";
        case HPRLPYBackend::RowTemplate: return "row-template";
        case HPRLPYBackend::AffineBlock: return "affine-block";
    }
    return "unknown";
}

bool is_signed_y_backend(HPRLPYBackend backend) {
    return backend == HPRLPYBackend::SignedUnitPacked ||
           backend == HPRLPYBackend::SignedUnitPackedCombined;
}

bool is_signed_x_backend(HPRLPXBackend backend) {
    return backend == HPRLPXBackend::SignedUnitPacked ||
           backend == HPRLPXBackend::SignedUnitSplitU16;
}

bool is_dictionary_x_backend(HPRLPXBackend backend) {
    return backend == HPRLPXBackend::PackedDictionary ||
           backend == HPRLPXBackend::FixedDegreePackedDictionary;
}

bool is_dictionary_y_backend(HPRLPYBackend backend) {
    return backend == HPRLPYBackend::PackedDictionary ||
           backend == HPRLPYBackend::FixedDegreePackedDictionary;
}

}
