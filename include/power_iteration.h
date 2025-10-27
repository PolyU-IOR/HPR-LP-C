#ifndef HPRLP_POWER_ITERATION_H
#define HPRLP_POWER_ITERATION_H

#include "structs.h"
#include "utils.h"
#include <random>

HPRLP_FLOAT power_method_cusparse(HPRLP_workspace_gpu *workspace, int max_iter = 50000, HPRLP_FLOAT tol = 1e-4);

#endif