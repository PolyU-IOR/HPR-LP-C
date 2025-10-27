#ifndef HPRLP_PREPROCESS_H
#define HPRLP_PREPROCESS_H

#include "structs.h"
#include "utils.h"

void copy_lpinfo_to_device(const LP_info_cpu *lp_info_cpu, LP_info_gpu *lp_info_gpu);

void allocate_memory(HPRLP_workspace_gpu *workspace, LP_info_gpu *lp_info_gpu);

void free_workspace(HPRLP_workspace_gpu *workspace);

void free_lp_info(LP_info_gpu *lp_info);

void free_lp_info_cpu(LP_info_cpu *lp_info);

#endif