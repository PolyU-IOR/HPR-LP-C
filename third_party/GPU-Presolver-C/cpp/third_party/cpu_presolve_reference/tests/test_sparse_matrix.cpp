#include "test_common.hpp"

#include "cpu_presolve/sparse_matrix.hpp"

using cpu_presolve::CscMatrix;
using cpu_presolve::CsrMatrix;

CPU_PRESOLVE_TEST(csr_matrix_rejects_bad_row_pointer_size) {
  bool threw = false;
  try {
    CsrMatrix matrix(2, 3, {0, 1}, {0}, {1.0});
    (void)matrix;
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CPU_PRESOLVE_REQUIRE(threw);
}

CPU_PRESOLVE_TEST(csc_to_csr_preserves_matrix_entries) {
  CscMatrix csc(
      3,
      4,
      {0, 2, 3, 3, 5},
      {0, 2, 1, 0, 2},
      {1.0, 5.0, 2.0, 3.0, 7.0});

  const CsrMatrix csr = csc.to_csr();

  CPU_PRESOLVE_REQUIRE(csr.rows() == 3);
  CPU_PRESOLVE_REQUIRE(csr.cols() == 4);
  CPU_PRESOLVE_REQUIRE(csr.row_ptr() == std::vector<int>({0, 2, 3, 5}));
  CPU_PRESOLVE_REQUIRE(csr.col_idx() == std::vector<int>({0, 3, 1, 0, 3}));
  CPU_PRESOLVE_REQUIRE(csr.values() == std::vector<double>({1.0, 3.0, 2.0, 5.0, 7.0}));
}

CPU_PRESOLVE_TEST(csr_to_csc_preserves_matrix_entries) {
  CsrMatrix csr(
      3,
      4,
      {0, 2, 3, 5},
      {0, 3, 1, 0, 3},
      {1.0, 3.0, 2.0, 5.0, 7.0});

  const CscMatrix csc = csr.to_csc();

  CPU_PRESOLVE_REQUIRE(csc.rows() == 3);
  CPU_PRESOLVE_REQUIRE(csc.cols() == 4);
  CPU_PRESOLVE_REQUIRE(csc.col_ptr() == std::vector<int>({0, 2, 3, 3, 5}));
  CPU_PRESOLVE_REQUIRE(csc.row_idx() == std::vector<int>({0, 2, 1, 0, 2}));
  CPU_PRESOLVE_REQUIRE(csc.values() == std::vector<double>({1.0, 5.0, 2.0, 3.0, 7.0}));
}
