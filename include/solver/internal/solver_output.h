#ifndef HPRLP_INTERNAL_SOLVER_OUTPUT_H
#define HPRLP_INTERNAL_SOLVER_OUTPUT_H

#include "api/structs.h"

#include <memory>
#include <streambuf>

struct HPRLP_finite_abs_range {
    bool has_value = false;
    bool has_negative_infinity = false;
    bool has_positive_infinity = false;
    HPRLP_FLOAT minimum = std::numeric_limits<HPRLP_FLOAT>::infinity();
    HPRLP_FLOAT maximum = 0.0;
};

struct HPRLP_numerical_ranges {
    HPRLP_finite_abs_range A;
    HPRLP_finite_abs_range AL;
    HPRLP_finite_abs_range AU;
    HPRLP_finite_abs_range l;
    HPRLP_finite_abs_range u;
    HPRLP_finite_abs_range c;
};

void print_solver_banner();
void print_solver_parameters(const HPRLP_parameters *param);
void print_solution_summary(const HPRLP_results &result,
                            HPRLP_FLOAT primal_objective,
                            HPRLP_FLOAT dual_objective,
                            HPRLP_FLOAT objective_gap,
                            HPRLP_FLOAT primal_residual,
                            HPRLP_FLOAT dual_residual);
void print_numerical_ranges(const LP_info_cpu *model, const char *stage);
void print_numerical_ranges(const HPRLP_numerical_ranges &ranges,
                            const char *stage);

// Filters informational stdout in normal mode while retaining the public
// progress table, compact summary, dimensions, and errors.  Debug mode is a
// zero-overhead pass-through.
class HPRLP_scoped_output_filter {
public:
    explicit HPRLP_scoped_output_filter(bool print_debug_info);
    ~HPRLP_scoped_output_filter();
    HPRLP_scoped_output_filter(const HPRLP_scoped_output_filter &) = delete;
    HPRLP_scoped_output_filter &operator=(const HPRLP_scoped_output_filter &) = delete;

private:
    std::streambuf *original_ = nullptr;
    std::unique_ptr<std::streambuf> filter_;
};

#endif
