#ifndef HPRLP_BATCHED_SOLVER_H
#define HPRLP_BATCHED_SOLVER_H

#include "structs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve B LPs with the same sparse matrix A and different vectors.
 *
 * Each batch member solves
 *   minimize    c_k' x + obj_constants[k]
 *   subject to  AL_k <= A x <= AU_k
 *               l_k <= x <= u_k
 *
 * All dense inputs are column-major:
 *   C, l, u are n x batch_size; AL, AU are m x batch_size.
 * This matches Julia/MATLAB storage and can be supplied from Python with
 * Fortran-contiguous NumPy arrays.
 */
HPRLP_batched_results solve_batched(const LP_info_cpu *model,
                                    int batch_size,
                                    const HPRLP_FLOAT *C,
                                    const HPRLP_FLOAT *AL,
                                    const HPRLP_FLOAT *AU,
                                    const HPRLP_FLOAT *l,
                                    const HPRLP_FLOAT *u,
                                    const HPRLP_FLOAT *obj_constants,
                                    const HPRLP_parameters *param);

void free_batched_results(HPRLP_batched_results *results);

#ifdef __cplusplus
}
#endif

#endif
