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

#ifndef HPRLP_NUM_THREADS
#define HPRLP_NUM_THREADS 256
#endif
#ifndef HPRLP_NUM_BLOCKS
#define HPRLP_NUM_BLOCKS(n) (((n) + HPRLP_NUM_THREADS - 1) / HPRLP_NUM_THREADS)
#endif

#endif
