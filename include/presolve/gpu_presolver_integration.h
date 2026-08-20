#ifndef HPRLP_GPU_PRESOLVER_INTEGRATION_H
#define HPRLP_GPU_PRESOLVER_INTEGRATION_H

#include "HPRLP.h"

#include <chrono>

bool run_embedded_gpu_presolve(const LP_info_cpu *model,
                               const HPRLP_parameters *param,
                               LP_info_cpu *reduced_model,
                               void **presolver_handle_out,
                               HPRLP_FLOAT *presolve_time_out,
                               HPRLP_FLOAT *folding_time_out);

bool run_embedded_gpu_presolve_device(const LP_info_cpu *model,
                                      const HPRLP_parameters *param,
                                      LP_info_gpu *reduced_model,
                                      void **presolver_handle_out,
                                      HPRLP_FLOAT *presolve_time_out,
                                      HPRLP_FLOAT *folding_time_out,
                                      std::chrono::steady_clock::time_point
                                          *first_device_ready_out);

bool apply_embedded_gpu_postsolve(HPRLP_results *result,
                                  void *presolver_handle,
                                  int original_m,
                                  int original_n);

bool gpu_postsolve_and_validate_original_kkt(HPRLP_results *result,
                                             const LP_info_cpu *original_model,
                                             void *presolver_handle,
                                             const HPRLP_parameters *param);

void free_embedded_gpu_presolver(void *presolver_handle);

#endif
