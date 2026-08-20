#include "cuda_kernels/backends/structured/grid_slack_laplacian_kernels.cuh"

#include <cmath>

namespace {

__device__ __forceinline__ HPRLP_FLOAT project_x(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t type) {
    if (type == 0) return value;
    if (type == 1) return fmax(value, lower);
    if (type == 2) return fmin(value, upper);
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ HPRLP_FLOAT project_y_delta(
    HPRLP_FLOAT value, HPRLP_FLOAT lower, HPRLP_FLOAT upper,
    std::uint8_t type) {
    if (type == 0) return 0.0;
    if (type == 1) return fmax(lower - value, 0.0);
    if (type == 2) return fmin(upper - value, 0.0);
    return fmax(lower - value, fmin(upper - value, 0.0));
}

}  // namespace

__global__ void grid_slack_laplacian_update_x_kernel(
    HPRLP_FLOAT *x, HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *lower,
    const HPRLP_FLOAT *upper, const std::uint8_t *bound_type,
    const HPRLP_FLOAT *objective, const HPRLP_FLOAT *last_x,
    const HPRLP_FLOAT *y, const int *AT_row_ptr,
    const HPRLP_FLOAT *AT_value, HPRLPGridSlackLaplacianShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    HPRLP_FLOAT acc = 0.0;
    const int value_start = AT_row_ptr[col];
    int value_offset = 0;
    if (col < shape.grid_count) {
        int raw_grid_index;
        if (col < shape.interior_side) {
            raw_grid_index = col + 1;
        } else if (col < shape.grid_count - shape.interior_side) {
            raw_grid_index = col + 2;
        } else {
            raw_grid_index = col + 3;
        }
        const int gy = raw_grid_index / shape.grid_side;
        const int gx = raw_grid_index % shape.grid_side;
        const bool interior =
            gy > 0 && gy < shape.grid_side - 1 &&
            gx > 0 && gx < shape.grid_side - 1;

        if (interior) {
            const int q =
                (gy - 1) * shape.interior_side + gx - 1;
            acc = fma(AT_value[value_start + value_offset++], y[q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.interior_count + q], acc);
        } else if (gx == 0 || gx == shape.grid_side - 1) {
            const int q = 2 * (gy - 1) +
                (gx == shape.grid_side - 1 ? 1 : 0);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.vertical_negative_row0 + q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.vertical_positive_row0 + q], acc);
        } else {
            const int q = 2 * (gx - 1) +
                (gy == shape.grid_side - 1 ? 1 : 0);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.horizontal_negative_row0 + q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.horizontal_positive_row0 + q], acc);
        }

        if (gy > 1 && gx > 0 && gx < shape.grid_side - 1) {
            const int q =
                (gy - 2) * shape.interior_side + gx - 1;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.laplacian_row0 + q], acc);
        }
        if (gy > 0 && gy < shape.grid_side - 1 && gx > 1) {
            const int q =
                (gy - 1) * shape.interior_side + gx - 2;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.laplacian_row0 + q], acc);
        }
        if (interior) {
            const int q =
                (gy - 1) * shape.interior_side + gx - 1;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.laplacian_row0 + q], acc);
        }
        if (gy > 0 && gy < shape.grid_side - 1 &&
            gx < shape.grid_side - 2) {
            const int q =
                (gy - 1) * shape.interior_side + gx;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.laplacian_row0 + q], acc);
        }
        if (gy < shape.grid_side - 2 && gx > 0 &&
            gx < shape.grid_side - 1) {
            const int q = gy * shape.interior_side + gx - 1;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.laplacian_row0 + q], acc);
        }
    } else {
        const int slack = col - shape.grid_count;
        if (slack < shape.interior_count) {
            acc = fma(AT_value[value_start + value_offset++],
                      y[slack], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.interior_count + slack], acc);
        } else {
            const int q = slack - shape.interior_count;
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.vertical_negative_row0 + q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.vertical_positive_row0 + q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.horizontal_negative_row0 + q], acc);
            acc = fma(AT_value[value_start + value_offset++],
                      y[shape.horizontal_positive_row0 + q], acc);
        }
    }

    const HPRLP_FLOAT xi = x[col];
    const HPRLP_FLOAT z_temp =
        fma(sigma_params[0], acc - objective[col], xi);
    const HPRLP_FLOAT x_bar = project_x(
        z_temp, lower[col], upper[col], bound_type[col]);
    const HPRLP_FLOAT x_hat_value = 2.0 * x_bar - xi;
    x[col] = fma(halpern_factors[1], x_hat_value,
                 halpern_factors[0] * last_x[col]);
    x_hat[col] = x_hat_value;
}

__global__ void grid_slack_laplacian_update_y_kernel(
    HPRLP_FLOAT *y, const HPRLP_FLOAT *lower, const HPRLP_FLOAT *upper,
    const std::uint8_t *bound_type, const HPRLP_FLOAT *last_y,
    const HPRLP_FLOAT *x_hat, const HPRLP_FLOAT *A_value,
    HPRLPGridSlackLaplacianShape shape,
    const HPRLP_FLOAT *sigma_params,
    const HPRLP_FLOAT *halpern_factors, int m) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    HPRLP_FLOAT acc = 0.0;
    const int value_start = row < shape.laplacian_row0
        ? 2 * row
        : 2 * shape.laplacian_row0 +
              5 * (row - shape.laplacian_row0);
    int value_offset = 0;
    if (row < shape.interior_count) {
        const int q = row;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_interior_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.grid_count + q], acc);
    } else if (row < shape.vertical_negative_row0) {
        const int q = row - shape.interior_count;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_interior_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.grid_count + q], acc);
    } else if (row < shape.vertical_positive_row0) {
        const int q = row - shape.vertical_negative_row0;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_vertical_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.boundary_slack0 + q], acc);
    } else if (row < shape.horizontal_negative_row0) {
        const int q = row - shape.vertical_positive_row0;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_vertical_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.boundary_slack0 + q], acc);
    } else if (row < shape.horizontal_positive_row0) {
        const int q = row - shape.horizontal_negative_row0;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_horizontal_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.boundary_slack0 + q], acc);
    } else if (row < shape.laplacian_row0) {
        const int q = row - shape.horizontal_positive_row0;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_horizontal_col(shape, q)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[shape.boundary_slack0 + q], acc);
    } else {
        const int q = row - shape.laplacian_row0;
        const int gy = 1 + q / shape.interior_side;
        const int gx = 1 + q % shape.interior_side;
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_grid_col(shape, gy - 1, gx)],
                  acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_grid_col(shape, gy, gx - 1)],
                  acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_grid_col(shape, gy, gx)], acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_grid_col(shape, gy, gx + 1)],
                  acc);
        acc = fma(A_value[value_start + value_offset++],
                  x_hat[hprlp_grid_slack_grid_col(shape, gy + 1, gx)],
                  acc);
    }

    const HPRLP_FLOAT yi = y[row];
    const HPRLP_FLOAT value = fma(-sigma_params[1], yi, acc);
    const HPRLP_FLOAT delta = project_y_delta(
        value, lower[row], upper[row], bound_type[row]);
    const HPRLP_FLOAT y_bar = sigma_params[2] * delta;
    const HPRLP_FLOAT y_hat = 2.0 * y_bar - yi;
    y[row] = fma(halpern_factors[1], y_hat,
                 halpern_factors[0] * last_y[row]);
}
