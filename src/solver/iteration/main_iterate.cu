#include "solver/iteration/main_iterate.h"
#include "solver/backends/autotune/autotune_probe_policy.h"
#include "solver/backends/signed_unit_launcher.cuh"
#include "solver/backends/packed_dictionary_launcher.cuh"
#include "solver/backends/unit_factorized_launcher.cuh"
#include "cuda_kernels/cuda_check.h"
#include "cuda_kernels/backends/structured/factorized_stencil_kernels.cuh"
#include "cuda_kernels/backends/structured/grid_slack_laplacian_kernels.cuh"
#include "cuda_kernels/backends/unit/signed_unit_kernels.cuh"
#include "gpu/preprocessing/operators/dictionary/packed_dictionary_operator.h"
#include "gpu/preprocessing/policies/row_bucket_policy.h"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>


#include "../backends/detail/backend_common.inc.cu"
#include "detail/runtime_state.inc.cu"

#include "../backends/detail/backend_state.inc.cu"
#include "../backends/xz/simple_cusparse_backend.inc.cu"
#include "../backends/xz/specialized/generic_fused_backends.inc.cu"
#include "../backends/xz/specialized/unit_backends.inc.cu"
#include "../backends/xz/specialized/dictionary_backends.inc.cu"
#include "../backends/xz/specialized/structured_backends.inc.cu"
#include "../backends/y/simple_cusparse_backend.inc.cu"
#include "../backends/y/specialized/generic_fused_backends.inc.cu"
#include "../backends/y/specialized/unit_backends.inc.cu"
#include "../backends/y/specialized/dictionary_backends.inc.cu"
#include "../backends/y/specialized/structured_backends.inc.cu"
#include "../backends/autotune/backend_selection.inc.cu"

#include "detail/residuals.inc.cu"
#include "detail/restart_control.inc.cu"
#include "../backends/xz/update_xz_classifier.inc.cu"
#include "../backends/y/update_y_classifier.inc.cu"
#include "detail/weighted_norm.inc.cu"
#include "../backends/autotune/backend_candidates.inc.cu"
#include "../backends/autotune/backend_diagnostics.inc.cu"
#include "../backends/autotune/backend_autotuner.inc.cu"
