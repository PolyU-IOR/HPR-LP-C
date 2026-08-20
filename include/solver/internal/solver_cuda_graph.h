#ifndef HPRLP_INTERNAL_SOLVER_CUDA_GRAPH_H
#define HPRLP_INTERNAL_SOLVER_CUDA_GRAPH_H

#include "api/structs.h"

void rebuild_cuda_graph(HPRLP_workspace_gpu *workspace);
void launch_check_cuda_graph(HPRLP_workspace_gpu *workspace);
void update_unit_coltile_backend_from_zero_density(
    HPRLP_workspace_gpu *workspace,
    const HPRLP_parameters *parameters,
    int completed_iteration);

#endif
