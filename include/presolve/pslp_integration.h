#ifndef HPRLP_PSLP_INTEGRATION_H
#define HPRLP_PSLP_INTEGRATION_H

#include "HPRLP.h"

bool run_embedded_pslp_presolve(const LP_info_cpu *model,
                                const HPRLP_parameters *param,
                                LP_info_cpu *reduced_model,
                                void **presolver_handle_out);

bool apply_embedded_pslp_postsolve(HPRLP_results *result,
                                   void *presolver_handle,
                                   int original_m,
                                   int original_n);

bool postsolve_and_validate_original_kkt(HPRLP_results *result,
                                         const LP_info_cpu *original_model,
                                         void *presolver_handle,
                                         const HPRLP_parameters *param);

void free_embedded_pslp_presolver(void *presolver_handle);

#endif