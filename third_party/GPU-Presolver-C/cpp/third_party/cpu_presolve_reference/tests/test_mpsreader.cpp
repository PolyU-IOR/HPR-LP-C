#include "test_common.hpp"

#include "cpu_presolve/mpsreader.hpp"

#include <cmath>
#include <limits>
#include <sstream>

using cpu_presolve::read_mps;

CPU_PRESOLVE_TEST(mpsreader_parses_rows_columns_rhs_and_bounds) {
  const double inf = std::numeric_limits<double>::infinity();
  std::istringstream input(
      "NAME          toy\n"
      "ROWS\n"
      " N  COST\n"
      " E  EQ1\n"
      " G  GE1\n"
      " L  LE1\n"
      "COLUMNS\n"
      "    X1        COST       3.0   EQ1        2.0\n"
      "    X1        GE1        1.0\n"
      "    X2        COST       5.0   LE1       -4.0\n"
      "    X2        EQ1        1.0\n"
      "RHS\n"
      "    RHS1      EQ1       10.0   GE1        7.0\n"
      "    RHS1      LE1       20.0\n"
      "BOUNDS\n"
      " LO BND1      X1        -1.0\n"
      " UP BND1      X1         6.0\n"
      " FR BND1      X2\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  CPU_PRESOLVE_REQUIRE(model.name == "toy");
  CPU_PRESOLVE_REQUIRE(lp.num_rows() == 3);
  CPU_PRESOLVE_REQUIRE(lp.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(lp.objective() == std::vector<double>({3.0, 5.0}));
  CPU_PRESOLVE_REQUIRE(lp.row_lower() == std::vector<double>({10.0, 7.0, -inf}));
  CPU_PRESOLVE_REQUIRE(lp.row_upper() == std::vector<double>({10.0, inf, 20.0}));
  CPU_PRESOLVE_REQUIRE(lp.col_lower() == std::vector<double>({-1.0, -inf}));
  CPU_PRESOLVE_REQUIRE(lp.col_upper() == std::vector<double>({6.0, inf}));
  CPU_PRESOLVE_REQUIRE(lp.csc().col_ptr() == std::vector<int>({0, 2, 4}));
  CPU_PRESOLVE_REQUIRE(lp.csc().row_idx() == std::vector<int>({0, 1, 0, 2}));
  CPU_PRESOLVE_REQUIRE(lp.csc().values() == std::vector<double>({2.0, 1.0, 1.0, -4.0}));
}

CPU_PRESOLVE_TEST(mpsreader_supports_ranges_binary_bounds_and_duplicate_entries) {
  std::istringstream input(
      "NAME ranged\n"
      "ROWS\n"
      " N OBJ\n"
      " E EQ1\n"
      " L LE1\n"
      " G GE1\n"
      "COLUMNS\n"
      "    MARK0000  'MARKER'                 'INTORG'\n"
      "    X1        OBJ        1.0   EQ1        1.0\n"
      "    X1        LE1        2.0   LE1        3.0\n"
      "    MARK0001  'MARKER'                 'INTEND'\n"
      "    X2        OBJ        2.0   GE1       -1.0\n"
      "RHS\n"
      "    RHS1      EQ1       10.0   LE1       20.0\n"
      "    RHS1      GE1        5.0\n"
      "RANGES\n"
      "    RNG1      EQ1        4.0   LE1        6.0\n"
      "    RNG1      GE1        7.0\n"
      "BOUNDS\n"
      " BV BND1      X1\n"
      " UI BND1      X2         9.0\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  CPU_PRESOLVE_REQUIRE(model.name == "ranged");
  CPU_PRESOLVE_REQUIRE(lp.num_rows() == 3);
  CPU_PRESOLVE_REQUIRE(lp.num_cols() == 2);
  CPU_PRESOLVE_REQUIRE(lp.objective() == std::vector<double>({1.0, 2.0}));
  CPU_PRESOLVE_REQUIRE(lp.row_lower() == std::vector<double>({10.0, 14.0, 5.0}));
  CPU_PRESOLVE_REQUIRE(lp.row_upper() == std::vector<double>({14.0, 20.0, 12.0}));
  CPU_PRESOLVE_REQUIRE(lp.col_lower() == std::vector<double>({0.0, 0.0}));
  CPU_PRESOLVE_REQUIRE(lp.col_upper() == std::vector<double>({1.0, 9.0}));
  CPU_PRESOLVE_REQUIRE(lp.csc().col_ptr() == std::vector<int>({0, 2, 3}));
  CPU_PRESOLVE_REQUIRE(lp.csc().row_idx() == std::vector<int>({0, 1, 2}));
  CPU_PRESOLVE_REQUIRE(lp.csc().values() == std::vector<double>({1.0, 5.0, -1.0}));
}

CPU_PRESOLVE_TEST(mpsreader_matches_julia_decimal_rounding_for_micro_bounds) {
  std::istringstream input(
      "NAME micro\n"
      "ROWS\n"
      " N OBJ\n"
      " L ROW1\n"
      "COLUMNS\n"
      "    X1        OBJ        1.0   ROW1       1.0\n"
      "RHS\n"
      "    RHS1      ROW1       1.0\n"
      "BOUNDS\n"
      " UP BND1      X1         0.000001\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& upper = model.lp.col_upper();

  CPU_PRESOLVE_REQUIRE(upper.size() == 1);
  CPU_PRESOLVE_REQUIRE(upper[0] > 1.0e-6);
}

CPU_PRESOLVE_TEST(mpsreader_auto_falls_back_to_fixed_format) {
  std::istringstream input(
      "NAME          fixed\n"
      "ROWS\n"
      " N  COST\n"
      " L  ROW00001\n"
      "COLUMNS\n"
      "    COL00001  COST       1.0   ROW00001   2.0\n"
      "RHS\n"
      "    RHS00001  ROW00001   3.0\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  CPU_PRESOLVE_REQUIRE(model.name == "fixed");
  CPU_PRESOLVE_REQUIRE(lp.num_rows() == 1);
  CPU_PRESOLVE_REQUIRE(lp.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(lp.objective() == std::vector<double>({1.0}));
  CPU_PRESOLVE_REQUIRE(lp.row_upper() == std::vector<double>({3.0}));
  CPU_PRESOLVE_REQUIRE(lp.csc().values() == std::vector<double>({2.0}));
}

CPU_PRESOLVE_TEST(mpsreader_supports_fixed_format_continuation_cards) {
  std::istringstream input(
      "NAME          PLAN\n"
      "ROWS\n"
      " N  VALUE\n"
      " E  YIELD\n"
      " L  FE\n"
      " L  CU\n"
      "COLUMNS\n"
      "    BIN1      VALUE           .03000   YIELD          1.00000\n"
      "              FE              .15000   CU              .03000\n"
      "RHS\n"
      "    RHS1      YIELD          2.00000   FE              3.00000\n"
      "              CU             4.00000\n"
      "BOUNDS\n"
      " UP BND1      BIN1         200.00000\n"
      " LO           BIN1          10.00000\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  CPU_PRESOLVE_REQUIRE(model.name == "PLAN");
  CPU_PRESOLVE_REQUIRE(lp.num_rows() == 3);
  CPU_PRESOLVE_REQUIRE(lp.num_cols() == 1);
  CPU_PRESOLVE_REQUIRE(std::fabs(lp.objective()[0] - 0.03) < 1.0e-12);
  CPU_PRESOLVE_REQUIRE(lp.row_lower() == std::vector<double>({2.0, -std::numeric_limits<double>::infinity(),
                                                              -std::numeric_limits<double>::infinity()}));
  CPU_PRESOLVE_REQUIRE(lp.row_upper() == std::vector<double>({2.0, 3.0, 4.0}));
  CPU_PRESOLVE_REQUIRE(lp.col_lower() == std::vector<double>({10.0}));
  CPU_PRESOLVE_REQUIRE(lp.col_upper() == std::vector<double>({200.0}));
  CPU_PRESOLVE_REQUIRE(lp.csc().values().size() == 3);
  CPU_PRESOLVE_REQUIRE(std::fabs(lp.csc().values()[0] - 1.0) < 1.0e-12);
  CPU_PRESOLVE_REQUIRE(std::fabs(lp.csc().values()[1] - 0.15) < 1.0e-12);
  CPU_PRESOLVE_REQUIRE(std::fabs(lp.csc().values()[2] - 0.03) < 1.0e-12);
}
