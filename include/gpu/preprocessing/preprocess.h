#ifndef HPRLP_PREPROCESS_H
#define HPRLP_PREPROCESS_H

#include "api/structs.h"
#include "support/utils.h"

void copy_lpinfo_to_device(const LP_info_cpu *lp_info_cpu, LP_info_gpu *lp_info_gpu);

// Build the packed-dictionary metadata used by the normal-update autotuner
// directly from the device-resident presolved A and A^T matrices.
bool prepare_device_packed_dictionary_metadata(LP_info_gpu *lp_info_gpu);

// Recreate baseline unit/signed/state/col-tile metadata without downloading
// the presolved numerical arrays from the GPU.
bool prepare_device_operator_metadata(LP_info_gpu *lp_info_gpu);

bool build_stable_device_transpose(const sparseMatrix *matrix,
                                   sparseMatrix **transpose_out);

void allocate_memory(HPRLP_workspace_gpu *workspace, LP_info_gpu *lp_info_gpu);

void analyze_spmv_pattern(HPRLP_workspace_gpu *workspace, const HPRLP_parameters *param);

void prepare_unit_operators(HPRLP_workspace_gpu *workspace, const Scaling_info *scaling_info);

void free_workspace(HPRLP_workspace_gpu *workspace);

void free_lp_info(LP_info_gpu *lp_info);

void free_lp_info_cpu(LP_info_cpu *lp_info);

#endif
