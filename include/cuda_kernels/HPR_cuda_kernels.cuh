#ifndef HPRLP_CUDA_KERNELS_H
#define HPRLP_CUDA_KERNELS_H

// Compatibility umbrella for solver code that launches kernels from several
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>

// categories. New code should include the narrow category header it uses.
#include "shared/vector_kernels.cuh"
#include "shared/scaling_kernels.cuh"
#include "shared/residual_kernels.cuh"
#include "shared/halpern_kernels.cuh"

#include "backends/simple/simple_update_kernels.cuh"
#include "backends/generic/generic_fused_kernels.cuh"
#include "backends/unit/unit_kernels.cuh"
#include "backends/dictionary/dictionary_kernels.cuh"
#include "backends/structured/structured_kernels.cuh"

#ifndef numThreads
#define numThreads 256
#endif
#ifndef numBlocks
#define numBlocks(n) (((n) + numThreads - 1) / numThreads)
#endif

#endif
