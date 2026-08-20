#include "gpu_presolver/model/lp_problem.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  const gpu_presolver::model::CscMatrix matrix{
      3,
      4,
      {0, 2, 3, 3, 5},
      {0, 2, 1, 0, 2},
      {1.0, 3.0, -2.0, 4.0, 5.0},
  };

  const gpu_presolver::model::LpProblem problem{
      matrix,
      {1.0, 2.0, 3.0, 4.0},
      {0.0, -1.0, 2.0},
      {10.0, 4.0, 8.0},
      {0.0, 0.0, 0.0, 0.0},
      {100.0, 100.0, 100.0, 100.0},
      7.5,
  };

  assert(problem.rows() == 3);
  assert(problem.cols() == 4);
  assert(problem.nnz() == 5);
  assert(problem.objective_constant() == 7.5);

  std::cout << "test_lp_problem passed\n";
  return 0;
}

