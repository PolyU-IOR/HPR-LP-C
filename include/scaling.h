#ifndef HPRLP_SCALING_H
#define HPRLP_SCALING_H

#include "utils.h"
#include "structs.h"

void scaling(LP_info_gpu *lp_info_gpu, Scaling_info* scaling_info, const HPRLP_parameters *param, cublasHandle_t cublasHandle);

void free_scaling_info(Scaling_info *scaling_info);

#endif