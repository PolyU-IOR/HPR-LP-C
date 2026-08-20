#include "cpu_presolve/presolver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace cpu_presolve {

PresolveResult::PresolveResult(LpModel reduced_model,
                               PresolveStatus status,
                               bool changed,
                               std::vector<FixedColumnRecord> fixed_columns,
                               double objective_shift)
    : reduced_model_(std::move(reduced_model)),
      status_(status),
      changed_(changed),
      fixed_columns_(std::move(fixed_columns)),
      objective_shift_(objective_shift) {}

namespace {

struct SparsePatternSignature {
  int index = 0;
  int length = 0;
  std::uint64_t hash = 0;
  std::uint64_t value_hash = 0;
};

std::uint64_t mix_hash(std::uint64_t seed, std::uint64_t value) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
  return seed * kPrime;
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t julia_parallel_hash_mix(std::uint64_t hash, std::uint64_t value) {
  return (hash ^ value) * 0x100000001b3ULL;
}

std::uint64_t hash_int_sequence(const std::vector<int>& values, int begin, int end) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (int pos = begin; pos < end; ++pos) {
    hash = mix_hash(hash, static_cast<std::uint64_t>(values[static_cast<std::size_t>(pos)]));
  }
  return hash;
}

std::uint64_t hash_scaled_value_sequence(const std::vector<double>& values, int begin, int end) {
  double scale = 0.0;
  double sign = 1.0;
  bool sign_set = false;
  for (int pos = begin; pos < end; ++pos) {
    const double value = values[static_cast<std::size_t>(pos)];
    scale = std::max(scale, std::fabs(value));
    if (!sign_set && std::fabs(value) > 0.0) {
      sign = value < 0.0 ? -1.0 : 1.0;
      sign_set = true;
    }
  }
  if (scale == 0.0) {
    return 0;
  }

  std::uint64_t hash = 1469598103934665603ULL;
  for (int pos = begin; pos < end; ++pos) {
    const double normalized = sign * values[static_cast<std::size_t>(pos)] / scale;
    const auto quantized = static_cast<std::int64_t>(std::llround(normalized * 1.0e9));
    hash = mix_hash(hash, static_cast<std::uint64_t>(quantized));
  }
  return hash;
}

std::uint64_t hash_julia_parallel_col_sequence(const std::vector<int>& row_idx,
                                               const std::vector<double>& values,
                                               int begin,
                                               int end) {
  if (begin >= end) {
    return julia_parallel_hash_mix(0xcbf29ce484222325ULL, 0);
  }
  const double pivot = values[static_cast<std::size_t>(begin)];
  if (std::fabs(pivot) <= 0.0) {
    return julia_parallel_hash_mix(0xcbf29ce484222325ULL, 0);
  }

  std::uint64_t hash = julia_parallel_hash_mix(
      0xcbf29ce484222325ULL,
      static_cast<std::uint64_t>(end - begin));
  for (int pos = begin; pos < end; ++pos) {
    hash = julia_parallel_hash_mix(
        hash,
        static_cast<std::uint64_t>(row_idx[static_cast<std::size_t>(pos)]));
    hash = julia_parallel_hash_mix(
        hash,
        double_bits(values[static_cast<std::size_t>(pos)] / pivot));
  }
  return hash;
}

bool signature_less(const SparsePatternSignature& left, const SparsePatternSignature& right) {
  if (left.length != right.length) {
    return left.length < right.length;
  }
  if (left.hash != right.hash) {
    return left.hash < right.hash;
  }
  if (left.value_hash != right.value_hash) {
    return left.value_hash < right.value_hash;
  }
  return left.index < right.index;
}

std::vector<SparsePatternSignature> row_pattern_signatures(const CsrMatrix& csr) {
  std::vector<SparsePatternSignature> signatures;
  signatures.reserve(static_cast<std::size_t>(csr.rows()));
  for (int row = 0; row < csr.rows(); ++row) {
    const int begin = csr.row_ptr()[row];
    const int end = csr.row_ptr()[row + 1];
    const int length = end - begin;
    if (length <= 0) {
      continue;
    }
    signatures.push_back(SparsePatternSignature{
        row,
        length,
        hash_int_sequence(csr.col_idx(), begin, end),
        hash_scaled_value_sequence(csr.values(), begin, end)});
  }
  std::sort(signatures.begin(), signatures.end(), signature_less);
  return signatures;
}

std::vector<SparsePatternSignature> col_pattern_signatures(const CscMatrix& csc) {
  std::vector<SparsePatternSignature> signatures;
  signatures.reserve(static_cast<std::size_t>(csc.cols()));
  for (int col = 0; col < csc.cols(); ++col) {
    const int begin = csc.col_ptr()[col];
    const int end = csc.col_ptr()[col + 1];
    const int length = end - begin;
    if (length <= 0) {
      continue;
    }
    signatures.push_back(SparsePatternSignature{
        col,
        length,
        hash_julia_parallel_col_sequence(csc.row_idx(), csc.values(), begin, end),
        0});
  }
  std::sort(signatures.begin(), signatures.end(), signature_less);
  return signatures;
}

bool is_close_fixed_column(double lower, double upper, double tolerance) {
  return std::isfinite(lower) && std::isfinite(upper) && std::fabs(upper - lower) <= tolerance;
}

PresolveResult apply_close_bounds(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  std::vector<unsigned char> fixed(static_cast<std::size_t>(n), 0);
  std::vector<double> fixed_value(static_cast<std::size_t>(n), 0.0);
  std::vector<double> row_shift(static_cast<std::size_t>(m), 0.0);
  std::vector<FixedColumnRecord> fixed_columns;
  double objective_shift = 0.0;

  const CscMatrix& csc = model.csc();
  for (int col = 0; col < n; ++col) {
    const double lower = model.col_lower()[col];
    const double upper = model.col_upper()[col];
    if (!is_close_fixed_column(lower, upper, options.bound_tolerance)) {
      continue;
    }

    const double value = 0.5 * (lower + upper);
    fixed[static_cast<std::size_t>(col)] = 1;
    fixed_value[static_cast<std::size_t>(col)] = value;
    objective_shift += model.objective()[col] * value;
    fixed_columns.push_back(FixedColumnRecord{col, value, model.objective()[col]});

    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int row = csc.row_idx()[pos];
      row_shift[static_cast<std::size_t>(row)] += csc.values()[pos] * value;
    }
  }

  if (fixed_columns.empty()) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  for (int row = 0; row < m; ++row) {
    new_row_lower[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
    new_row_upper[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
  }

  int kept_cols = 0;
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  std::vector<double> new_objective;
  std::vector<double> new_col_lower;
  std::vector<double> new_col_upper;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    if (fixed[static_cast<std::size_t>(col)] != 0) {
      continue;
    }

    ++kept_cols;
    new_objective.push_back(model.objective()[col]);
    new_col_lower.push_back(model.col_lower()[col]);
    new_col_upper.push_back(model.col_upper()[col]);
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      new_row_idx.push_back(csc.row_idx()[pos]);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m, kept_cols, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      std::move(new_objective),
      std::move(new_row_lower),
      std::move(new_row_upper),
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant() + objective_shift);

  return PresolveResult(
      std::move(reduced),
      PresolveStatus::kOk,
      true,
      std::move(fixed_columns),
      objective_shift);
}

bool empty_row_is_feasible(double lower, double upper, double tolerance) {
  return lower <= tolerance && upper >= -tolerance;
}

bool empty_col_is_unbounded(double objective,
                            double lower,
                            double upper,
                            double zero_tolerance);

double empty_col_fixed_value(double objective,
                             double lower,
                             double upper,
                             double zero_tolerance);

PresolveResult apply_empty_rows(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CsrMatrix& csr = model.csr();

  std::vector<unsigned char> keep_row(static_cast<std::size_t>(m), 1);
  int removed_rows = 0;
  for (int row = 0; row < m; ++row) {
    const bool is_empty = csr.row_ptr()[row] == csr.row_ptr()[row + 1];
    if (!is_empty) {
      continue;
    }

    if (!empty_row_is_feasible(
            model.row_lower()[row], model.row_upper()[row], options.feasibility_tolerance)) {
      return PresolveResult(model, PresolveStatus::kInfeasible, false, {}, 0.0);
    }

    keep_row[static_cast<std::size_t>(row)] = 0;
    ++removed_rows;
  }

  if (removed_rows == 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<int> old_to_new_row(static_cast<std::size_t>(m), -1);
  std::vector<double> new_row_lower;
  std::vector<double> new_row_upper;
  new_row_lower.reserve(static_cast<std::size_t>(m - removed_rows));
  new_row_upper.reserve(static_cast<std::size_t>(m - removed_rows));

  int next_row = 0;
  for (int row = 0; row < m; ++row) {
    if (keep_row[static_cast<std::size_t>(row)] == 0) {
      continue;
    }
    old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
    new_row_lower.push_back(model.row_lower()[row]);
    new_row_upper.push_back(model.row_upper()[row]);
  }

  const CscMatrix& csc = model.csc();
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int old_row = csc.row_idx()[pos];
      const int new_row = old_to_new_row[static_cast<std::size_t>(old_row)];
      if (new_row < 0) {
        continue;
      }
      new_row_idx.push_back(new_row);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m - removed_rows, n, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      model.objective(),
      std::move(new_row_lower),
      std::move(new_row_upper),
      model.col_lower(),
      model.col_upper(),
      model.obj_constant());

  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

PresolveResult apply_singleton_rows(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CsrMatrix& csr = model.csr();

  std::vector<unsigned char> remove_row(static_cast<std::size_t>(m), 0);
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();
  int removed_rows = 0;

  for (int row = 0; row < m; ++row) {
    const int row_start = csr.row_ptr()[row];
    const int row_stop = csr.row_ptr()[row + 1];
    if (row_stop - row_start != 1) {
      continue;
    }

    const int col = csr.col_idx()[row_start];
    const double a = csr.values()[row_start];
    if (std::fabs(a) <= options.zero_tolerance) {
      continue;
    }

    const double lower = model.row_lower()[row];
    const double upper = model.row_upper()[row];
    const double implied_lower = a > 0.0 ? lower / a : upper / a;
    const double implied_upper = a > 0.0 ? upper / a : lower / a;
    if (implied_lower > new_col_lower[static_cast<std::size_t>(col)]) {
      new_col_lower[static_cast<std::size_t>(col)] = implied_lower;
    }
    if (implied_upper < new_col_upper[static_cast<std::size_t>(col)]) {
      new_col_upper[static_cast<std::size_t>(col)] = implied_upper;
    }
    remove_row[static_cast<std::size_t>(row)] = 1;
    ++removed_rows;
  }

  for (int col = 0; col < n; ++col) {
    if (new_col_lower[static_cast<std::size_t>(col)] >
        new_col_upper[static_cast<std::size_t>(col)] + options.bound_tolerance) {
      return PresolveResult(model, PresolveStatus::kInfeasible, false, {}, 0.0);
    }
  }

  bool bound_changed = false;
  for (int col = 0; col < n; ++col) {
    if (new_col_lower[static_cast<std::size_t>(col)] != model.col_lower()[col] ||
        new_col_upper[static_cast<std::size_t>(col)] != model.col_upper()[col]) {
      bound_changed = true;
      break;
    }
  }

  if (removed_rows == 0 && !bound_changed) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<int> old_to_new_row(static_cast<std::size_t>(m), -1);
  std::vector<double> new_row_lower;
  std::vector<double> new_row_upper;
  new_row_lower.reserve(static_cast<std::size_t>(m - removed_rows));
  new_row_upper.reserve(static_cast<std::size_t>(m - removed_rows));

  int next_row = 0;
  for (int row = 0; row < m; ++row) {
    if (remove_row[static_cast<std::size_t>(row)] != 0) {
      continue;
    }
    old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
    new_row_lower.push_back(model.row_lower()[row]);
    new_row_upper.push_back(model.row_upper()[row]);
  }

  const CscMatrix& csc = model.csc();
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int old_row = csc.row_idx()[pos];
      const int new_row = old_to_new_row[static_cast<std::size_t>(old_row)];
      if (new_row < 0) {
        continue;
      }
      new_row_idx.push_back(new_row);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m - removed_rows, n, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      model.objective(),
      std::move(new_row_lower),
      std::move(new_row_upper),
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant());

  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

double activity_product(double coefficient, double bound) {
  if (coefficient == 0.0) {
    return 0.0;
  }
  return coefficient * bound;
}

double term_min(double coefficient, double lower, double upper) {
  return coefficient >= 0.0 ? activity_product(coefficient, lower)
                            : activity_product(coefficient, upper);
}

double term_max(double coefficient, double lower, double upper) {
  return coefficient >= 0.0 ? activity_product(coefficient, upper)
                            : activity_product(coefficient, lower);
}

struct ActivitySum {
  double finite = 0.0;
  int negative_infinity = 0;
  int positive_infinity = 0;
  bool has_nan = false;
};

void add_activity_value(ActivitySum& sum, double value) {
  if (std::isnan(value)) {
    sum.has_nan = true;
  } else if (std::isinf(value)) {
    if (value < 0.0) {
      ++sum.negative_infinity;
    } else {
      ++sum.positive_infinity;
    }
  } else {
    sum.finite += value;
  }
}

bool finite_activity_excluding(const ActivitySum& sum, double excluded, double& value) {
  if (sum.has_nan || std::isnan(excluded)) {
    return false;
  }

  int negative_infinity = sum.negative_infinity;
  int positive_infinity = sum.positive_infinity;
  double finite = sum.finite;
  if (std::isinf(excluded)) {
    if (excluded < 0.0) {
      --negative_infinity;
    } else {
      --positive_infinity;
    }
  } else {
    finite -= excluded;
  }

  if (negative_infinity != 0 || positive_infinity != 0) {
    return false;
  }
  value = finite;
  return true;
}

bool activity_value_excluding(const ActivitySum& sum, double excluded, double& value) {
  if (sum.has_nan || std::isnan(excluded)) {
    return false;
  }

  ActivitySum adjusted = sum;
  if (std::isinf(excluded)) {
    if (excluded < 0.0) {
      --adjusted.negative_infinity;
    } else {
      --adjusted.positive_infinity;
    }
  } else {
    adjusted.finite -= excluded;
  }
  if (adjusted.negative_infinity < 0 || adjusted.positive_infinity < 0) {
    return false;
  }
  if (adjusted.negative_infinity != 0 && adjusted.positive_infinity != 0) {
    return false;
  }
  if (adjusted.negative_infinity != 0) {
    value = -std::numeric_limits<double>::infinity();
    return true;
  }
  if (adjusted.positive_infinity != 0) {
    value = std::numeric_limits<double>::infinity();
    return true;
  }
  value = adjusted.finite;
  return true;
}

bool activity_sum_value(const ActivitySum& sum, double& value) {
  if (sum.has_nan) {
    return false;
  }
  if (sum.negative_infinity != 0 && sum.positive_infinity != 0) {
    return false;
  }
  if (sum.negative_infinity != 0) {
    value = -std::numeric_limits<double>::infinity();
    return true;
  }
  if (sum.positive_infinity != 0) {
    value = std::numeric_limits<double>::infinity();
    return true;
  }
  value = sum.finite;
  return true;
}

struct CachedRowActivity {
  ActivitySum minimum;
  ActivitySum maximum;
};

struct RowActivitySums {
  std::vector<ActivitySum> minimum;
  std::vector<ActivitySum> maximum;
};

RowActivitySums compute_row_activity_sums(const LpModel& model) {
  const CsrMatrix& csr = model.csr();
  RowActivitySums sums;
  sums.minimum.resize(static_cast<std::size_t>(model.num_rows()));
  sums.maximum.resize(static_cast<std::size_t>(model.num_rows()));

  for (int row = 0; row < model.num_rows(); ++row) {
    for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
      const int col = csr.col_idx()[pos];
      const double a = csr.values()[pos];
      add_activity_value(
          sums.minimum[static_cast<std::size_t>(row)],
          term_min(a, model.col_lower()[col], model.col_upper()[col]));
      add_activity_value(
          sums.maximum[static_cast<std::size_t>(row)],
          term_max(a, model.col_lower()[col], model.col_upper()[col]));
    }
  }

  return sums;
}

using RowEntries = std::map<int, double>;

void add_row_entry(RowEntries& entries, int col, double value, double zero_tolerance) {
  if (std::fabs(value) <= zero_tolerance) {
    return;
  }
  entries[col] += value;
  if (std::fabs(entries[col]) <= zero_tolerance) {
    entries.erase(col);
  }
}

LpModel make_model_from_row_entries(int num_rows,
                                    int num_cols,
                                    const std::vector<RowEntries>& row_entries,
                                    std::vector<double> objective,
                                    std::vector<double> row_lower,
                                    std::vector<double> row_upper,
                                    std::vector<double> col_lower,
                                    std::vector<double> col_upper,
                                    double obj_constant,
                                    double zero_tolerance) {
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;
  row_ptr.reserve(static_cast<std::size_t>(num_rows + 1));
  row_ptr.push_back(0);

  for (const RowEntries& entries : row_entries) {
    for (const auto& entry : entries) {
      if (std::fabs(entry.second) <= zero_tolerance) {
        continue;
      }
      col_idx.push_back(entry.first);
      values.push_back(entry.second);
    }
    row_ptr.push_back(static_cast<int>(values.size()));
  }

  CsrMatrix csr(num_rows, num_cols, std::move(row_ptr), std::move(col_idx), std::move(values));
  CscMatrix csc = csr.to_csc();
  return LpModel(
      std::move(csc),
      std::move(csr),
      std::move(objective),
      std::move(row_lower),
      std::move(row_upper),
      std::move(col_lower),
      std::move(col_upper),
      obj_constant);
}

bool is_equality_row(double lower, double upper, double tolerance) {
  return std::isfinite(lower) && std::isfinite(upper) && std::fabs(upper - lower) <= tolerance;
}

bool singleton_column_support(const CscMatrix& csc,
                              int col,
                              double zero_tolerance,
                              int& support_row,
                              double& support_value) {
  support_row = -1;
  support_value = 0.0;
  int count = 0;
  for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
    const double value = csc.values()[pos];
    if (std::fabs(value) <= zero_tolerance) {
      continue;
    }
    ++count;
    support_row = csc.row_idx()[pos];
    support_value = value;
    if (count > 1) {
      return false;
    }
  }
  return count == 1;
}

int row_live_nnz(const CsrMatrix& csr, int row, double zero_tolerance) {
  int count = 0;
  for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
    if (std::fabs(csr.values()[pos]) > zero_tolerance) {
      ++count;
    }
  }
  return count;
}

bool row_activity_bounds_excluding(const LpModel& model,
                                   int row,
                                   int excluded_col,
                                   double& rest_min,
                                   double& rest_max) {
  rest_min = 0.0;
  rest_max = 0.0;
  const CsrMatrix& csr = model.csr();
  for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
    const int col = csr.col_idx()[pos];
    if (col == excluded_col) {
      continue;
    }
    const double lower = model.col_lower()[col];
    const double upper = model.col_upper()[col];
    const double min_term = term_min(csr.values()[pos], lower, upper);
    const double max_term = term_max(csr.values()[pos], lower, upper);
    if (std::isnan(min_term) || std::isnan(max_term)) {
      return false;
    }
    rest_min += min_term;
    rest_max += max_term;
    if (std::isnan(rest_min) || std::isnan(rest_max)) {
      return false;
    }
  }
  return true;
}

PresolveResult apply_activity_checks(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CsrMatrix& csr = model.csr();

  std::vector<unsigned char> remove_row(static_cast<std::size_t>(m), 0);
  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  int removed_rows = 0;
  bool side_changed = false;

  for (int row = 0; row < m; ++row) {
    const int row_start = csr.row_ptr()[row];
    const int row_stop = csr.row_ptr()[row + 1];
    const int row_nnz = row_stop - row_start;
    if (row_nnz <= 1) {
      continue;
    }

    const double lower = model.row_lower()[row];
    const double upper = model.row_upper()[row];
    const bool lower_finite = std::isfinite(lower);
    const bool upper_finite = std::isfinite(upper);
    if (lower_finite && upper_finite && std::fabs(upper - lower) <= options.bound_tolerance) {
      continue;
    }

    double row_min = 0.0;
    double row_max = 0.0;
    for (int pos = row_start; pos < row_stop; ++pos) {
      const int col = csr.col_idx()[pos];
      const double a = csr.values()[pos];
      if (a >= 0.0) {
        row_min += activity_product(a, model.col_lower()[col]);
        row_max += activity_product(a, model.col_upper()[col]);
      } else {
        row_min += activity_product(a, model.col_upper()[col]);
        row_max += activity_product(a, model.col_lower()[col]);
      }
    }

    const bool infeasible =
        (lower_finite && row_max < lower - options.bound_tolerance) ||
        (upper_finite && row_min > upper + options.bound_tolerance);
    if (infeasible) {
      return PresolveResult(model, PresolveStatus::kInfeasible, false, {}, 0.0);
    }

    const bool lower_implied = !lower_finite || row_min >= lower - options.bound_tolerance;
    const bool upper_implied = !upper_finite || row_max <= upper + options.bound_tolerance;
    if (lower_implied && upper_implied) {
      remove_row[static_cast<std::size_t>(row)] = 1;
      ++removed_rows;
    } else if (lower_finite && lower_implied) {
      new_row_lower[static_cast<std::size_t>(row)] = -std::numeric_limits<double>::infinity();
      side_changed = true;
    } else if (upper_finite && upper_implied) {
      new_row_upper[static_cast<std::size_t>(row)] = std::numeric_limits<double>::infinity();
      side_changed = true;
    }
  }

  if (removed_rows == 0 && !side_changed) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  if (removed_rows == 0) {
    LpModel reduced = model.with_row_bounds(
        std::move(new_row_lower),
        std::move(new_row_upper),
        model.obj_constant());
    return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
  }

  std::vector<int> old_to_new_row(static_cast<std::size_t>(m), -1);
  std::vector<double> kept_row_lower;
  std::vector<double> kept_row_upper;
  kept_row_lower.reserve(static_cast<std::size_t>(m - removed_rows));
  kept_row_upper.reserve(static_cast<std::size_t>(m - removed_rows));

  int next_row = 0;
  for (int row = 0; row < m; ++row) {
    if (remove_row[static_cast<std::size_t>(row)] != 0) {
      continue;
    }
    old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
    kept_row_lower.push_back(new_row_lower[static_cast<std::size_t>(row)]);
    kept_row_upper.push_back(new_row_upper[static_cast<std::size_t>(row)]);
  }

  const CscMatrix& csc = model.csc();
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int old_row = csc.row_idx()[pos];
      const int new_row = old_to_new_row[static_cast<std::size_t>(old_row)];
      if (new_row < 0) {
        continue;
      }
      new_row_idx.push_back(new_row);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m - removed_rows, n, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      model.objective(),
      std::move(kept_row_lower),
      std::move(kept_row_upper),
      model.col_lower(),
      model.col_upper(),
      model.obj_constant());

  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

PresolveResult apply_primal_propagation(const LpModel& model, const PresolveOptions& options) {
  const int n = model.num_cols();
  const CsrMatrix& csr = model.csr();
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();

  for (int row = 0; row < model.num_rows(); ++row) {
    const int row_start = csr.row_ptr()[row];
    const int row_stop = csr.row_ptr()[row + 1];
    if (row_stop <= row_start) {
      continue;
    }

    const double row_lower = model.row_lower()[row];
    const double row_upper = model.row_upper()[row];
    if (!std::isfinite(row_lower) && !std::isfinite(row_upper)) {
      continue;
    }

    ActivitySum row_min_sum;
    ActivitySum row_max_sum;
    for (int pos = row_start; pos < row_stop; ++pos) {
      const int col = csr.col_idx()[pos];
      const double a = csr.values()[pos];
      add_activity_value(row_min_sum, term_min(a, model.col_lower()[col], model.col_upper()[col]));
      add_activity_value(row_max_sum, term_max(a, model.col_lower()[col], model.col_upper()[col]));
    }

    for (int pos = row_start; pos < row_stop; ++pos) {
      const int col = csr.col_idx()[pos];
      const double a = csr.values()[pos];
      if (std::fabs(a) <= options.zero_tolerance) {
        continue;
      }

      double rest_min = 0.0;
      double rest_max = 0.0;
      const double current_min = term_min(a, model.col_lower()[col], model.col_upper()[col]);
      const double current_max = term_max(a, model.col_lower()[col], model.col_upper()[col]);
      const bool rest_min_finite = finite_activity_excluding(row_min_sum, current_min, rest_min);
      const bool rest_max_finite = finite_activity_excluding(row_max_sum, current_max, rest_max);

      if (a > 0.0) {
        if (std::isfinite(row_lower) && rest_max_finite) {
          const double implied_lower = (row_lower - rest_max) / a;
          if (implied_lower > new_col_lower[static_cast<std::size_t>(col)]) {
            new_col_lower[static_cast<std::size_t>(col)] = implied_lower;
          }
        }
        if (std::isfinite(row_upper) && rest_min_finite) {
          const double implied_upper = (row_upper - rest_min) / a;
          if (implied_upper < new_col_upper[static_cast<std::size_t>(col)]) {
            new_col_upper[static_cast<std::size_t>(col)] = implied_upper;
          }
        }
      } else {
        if (std::isfinite(row_upper) && rest_min_finite) {
          const double implied_lower = (row_upper - rest_min) / a;
          if (implied_lower > new_col_lower[static_cast<std::size_t>(col)]) {
            new_col_lower[static_cast<std::size_t>(col)] = implied_lower;
          }
        }
        if (std::isfinite(row_lower) && rest_max_finite) {
          const double implied_upper = (row_lower - rest_max) / a;
          if (implied_upper < new_col_upper[static_cast<std::size_t>(col)]) {
            new_col_upper[static_cast<std::size_t>(col)] = implied_upper;
          }
        }
      }
    }
  }

  bool changed = false;
  for (int col = 0; col < n; ++col) {
    const double lower = new_col_lower[static_cast<std::size_t>(col)];
    const double upper = new_col_upper[static_cast<std::size_t>(col)];
    if (lower > upper + options.feasibility_tolerance) {
      return PresolveResult(model, PresolveStatus::kInfeasible, false, {}, 0.0);
    }
    if (lower != model.col_lower()[col] || upper != model.col_upper()[col]) {
      changed = true;
    }
  }

  if (!changed) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  LpModel reduced = model.with_col_bounds(
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant());
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

bool parallel_row_ratio(const CsrMatrix& csr,
                        int rep_row,
                        int row,
                        double tolerance,
                        double& ratio) {
  const int rep_start = csr.row_ptr()[rep_row];
  const int rep_stop = csr.row_ptr()[rep_row + 1];
  const int row_start = csr.row_ptr()[row];
  const int row_stop = csr.row_ptr()[row + 1];
  const int len = rep_stop - rep_start;
  if (len <= 0 || row_stop - row_start != len) {
    return false;
  }

  bool ratio_set = false;
  ratio = 0.0;
  for (int offset = 0; offset < len; ++offset) {
    const int rep_pos = rep_start + offset;
    const int row_pos = row_start + offset;
    if (csr.col_idx()[rep_pos] != csr.col_idx()[row_pos]) {
      return false;
    }
    const double rep_value = csr.values()[rep_pos];
    const double row_value = csr.values()[row_pos];
    if (std::fabs(row_value) <= tolerance) {
      return false;
    }
    if (!ratio_set) {
      ratio = rep_value / row_value;
      ratio_set = true;
    }
    if (std::fabs(rep_value - ratio * row_value) > tolerance) {
      return false;
    }
  }
  return ratio_set;
}

void scaled_interval(double lower,
                     double upper,
                     double ratio,
                     double& scaled_lower,
                     double& scaled_upper) {
  const double a = lower * ratio;
  const double b = upper * ratio;
  if (ratio >= 0.0) {
    scaled_lower = a;
    scaled_upper = b;
  } else {
    scaled_lower = b;
    scaled_upper = a;
  }
}

PresolveResult apply_parallel_rows(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CsrMatrix& csr = model.csr();
  std::vector<unsigned char> remove_row(static_cast<std::size_t>(m), 0);
  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  int removed_rows = 0;

  const std::vector<SparsePatternSignature> signatures = row_pattern_signatures(csr);
  for (std::size_t group_begin = 0; group_begin < signatures.size();) {
    std::size_t group_end = group_begin + 1;
    while (group_end < signatures.size() &&
           signatures[group_end].length == signatures[group_begin].length &&
           signatures[group_end].hash == signatures[group_begin].hash &&
           signatures[group_end].value_hash == signatures[group_begin].value_hash) {
      ++group_end;
    }

    if (group_end - group_begin < 2) {
      group_begin = group_end;
      continue;
    }

    std::size_t rep_pos = group_begin;
    while (rep_pos < group_end &&
           remove_row[static_cast<std::size_t>(signatures[rep_pos].index)] != 0) {
      ++rep_pos;
    }
    if (rep_pos == group_end) {
      group_begin = group_end;
      continue;
    }

    const int rep = signatures[rep_pos].index;
    for (std::size_t row_pos = rep_pos + 1; row_pos < group_end; ++row_pos) {
      const int row = signatures[row_pos].index;
      if (remove_row[static_cast<std::size_t>(row)] != 0) {
        continue;
      }
      double ratio = 0.0;
      if (!parallel_row_ratio(csr, rep, row, options.zero_tolerance, ratio)) {
        continue;
      }

      double scaled_lower = 0.0;
      double scaled_upper = 0.0;
      scaled_interval(
          model.row_lower()[row],
          model.row_upper()[row],
          ratio,
          scaled_lower,
          scaled_upper);

      const double merged_lower = std::max(new_row_lower[static_cast<std::size_t>(rep)], scaled_lower);
      const double merged_upper = std::min(new_row_upper[static_cast<std::size_t>(rep)], scaled_upper);
      if (merged_lower > merged_upper + options.feasibility_tolerance) {
        return PresolveResult(model, PresolveStatus::kInfeasible, false, {}, 0.0);
      }

      new_row_lower[static_cast<std::size_t>(rep)] = merged_lower;
      new_row_upper[static_cast<std::size_t>(rep)] = merged_upper;
      remove_row[static_cast<std::size_t>(row)] = 1;
      ++removed_rows;
    }
    group_begin = group_end;
  }

  if (removed_rows == 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<int> old_to_new_row(static_cast<std::size_t>(m), -1);
  std::vector<double> kept_row_lower;
  std::vector<double> kept_row_upper;
  kept_row_lower.reserve(static_cast<std::size_t>(m - removed_rows));
  kept_row_upper.reserve(static_cast<std::size_t>(m - removed_rows));

  int next_row = 0;
  for (int row = 0; row < m; ++row) {
    if (remove_row[static_cast<std::size_t>(row)] != 0) {
      continue;
    }
    old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
    kept_row_lower.push_back(new_row_lower[static_cast<std::size_t>(row)]);
    kept_row_upper.push_back(new_row_upper[static_cast<std::size_t>(row)]);
  }

  const CscMatrix& csc = model.csc();
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);
  for (int col = 0; col < n; ++col) {
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int old_row = csc.row_idx()[pos];
      const int new_row = old_to_new_row[static_cast<std::size_t>(old_row)];
      if (new_row < 0) {
        continue;
      }
      new_row_idx.push_back(new_row);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m - removed_rows, n, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      model.objective(),
      std::move(kept_row_lower),
      std::move(kept_row_upper),
      model.col_lower(),
      model.col_upper(),
      model.obj_constant());
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

bool singleton_col_direct_unbounded(double objective,
                                    double coefficient,
                                    double row_lower,
                                    double row_upper,
                                    double col_lower,
                                    double col_upper,
                                    double zero_tolerance) {
  return ((objective > zero_tolerance && coefficient > zero_tolerance &&
           !std::isfinite(row_lower) && !std::isfinite(col_lower)) ||
          (objective > zero_tolerance && coefficient < -zero_tolerance &&
           !std::isfinite(row_upper) && !std::isfinite(col_lower)) ||
          (objective < -zero_tolerance && coefficient < -zero_tolerance &&
           !std::isfinite(row_lower) && !std::isfinite(col_upper)) ||
          (objective < -zero_tolerance && coefficient > zero_tolerance &&
           !std::isfinite(row_upper) && !std::isfinite(col_upper)));
}

bool singleton_col_eq_free_from_above(double implied_upper,
                                      double col_upper,
                                      double tolerance) {
  return !std::isfinite(col_upper) || implied_upper <= col_upper + tolerance;
}

bool singleton_col_eq_free_from_below(double implied_lower,
                                      double col_lower,
                                      double tolerance) {
  return !std::isfinite(col_lower) || implied_lower >= col_lower - tolerance;
}

bool singleton_col_implied_free_from_above(double coefficient,
                                           double row_lower,
                                           double row_upper,
                                           double col_upper,
                                           double rest_min,
                                           double rest_max,
                                           double tolerance) {
  if (!std::isfinite(col_upper)) {
    return true;
  }
  double implied_upper = std::numeric_limits<double>::infinity();
  if (coefficient > 0.0 && std::isfinite(row_upper)) {
    implied_upper = (row_upper - rest_min) / coefficient;
  } else if (coefficient < 0.0 && std::isfinite(row_lower)) {
    implied_upper = (row_lower - rest_max) / coefficient;
  }
  return implied_upper <= col_upper + tolerance;
}

bool singleton_col_implied_free_from_below(double coefficient,
                                           double row_lower,
                                           double row_upper,
                                           double col_lower,
                                           double rest_min,
                                           double rest_max,
                                           double tolerance) {
  if (!std::isfinite(col_lower)) {
    return true;
  }
  double implied_lower = -std::numeric_limits<double>::infinity();
  if (coefficient > 0.0 && std::isfinite(row_lower)) {
    implied_lower = (row_lower - rest_max) / coefficient;
  } else if (coefficient < 0.0 && std::isfinite(row_upper)) {
    implied_lower = (row_upper - rest_min) / coefficient;
  }
  return implied_lower >= col_lower - tolerance;
}

double singleton_col_active_side(double objective,
                                 double coefficient,
                                 double row_lower,
                                 double row_upper,
                                 double zero_tolerance) {
  if ((objective > zero_tolerance && coefficient > 0.0) ||
      (objective < -zero_tolerance && coefficient < 0.0)) {
    return row_lower;
  }
  if ((objective > zero_tolerance && coefficient < 0.0) ||
      (objective < -zero_tolerance && coefficient > 0.0)) {
    return row_upper;
  }
  return std::isfinite(row_lower) ? row_lower : row_upper;
}

PresolveResult apply_singleton_cols_dual_infer(const LpModel& model,
                                               const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();
  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  std::vector<unsigned char> row_changed(static_cast<std::size_t>(m), 0);
  bool changed = false;

  for (int col = 0; col < n; ++col) {
    int row = -1;
    double coefficient = 0.0;
    if (!singleton_column_support(csc, col, options.zero_tolerance, row, coefficient)) {
      continue;
    }
    if (row_changed[static_cast<std::size_t>(row)] != 0 ||
        row_live_nnz(model.csr(), row, options.zero_tolerance) <= 1 ||
        is_equality_row(model.row_lower()[row], model.row_upper()[row], options.bound_tolerance)) {
      continue;
    }

    const double objective = model.objective()[col];
    const double row_lower = model.row_lower()[row];
    const double row_upper = model.row_upper()[row];
    if (singleton_col_direct_unbounded(
            objective,
            coefficient,
            row_lower,
            row_upper,
            model.col_lower()[col],
            model.col_upper()[col],
            options.zero_tolerance)) {
      return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
    }

    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!row_activity_bounds_excluding(model, row, col, rest_min, rest_max)) {
      continue;
    }

    const bool free_above = singleton_col_implied_free_from_above(
        coefficient,
        row_lower,
        row_upper,
        model.col_upper()[col],
        rest_min,
        rest_max,
        options.bound_tolerance);
    const bool free_below = singleton_col_implied_free_from_below(
        coefficient,
        row_lower,
        row_upper,
        model.col_lower()[col],
        rest_min,
        rest_max,
        options.bound_tolerance);
    if (!free_above || !free_below) {
      continue;
    }

    const double active_side = singleton_col_active_side(
        objective,
        coefficient,
        row_lower,
        row_upper,
        options.zero_tolerance);
    if (!std::isfinite(active_side)) {
      continue;
    }

    new_row_lower[static_cast<std::size_t>(row)] = active_side;
    new_row_upper[static_cast<std::size_t>(row)] = active_side;
    row_changed[static_cast<std::size_t>(row)] = 1;
    changed = true;
  }

  if (!changed) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  LpModel reduced(
      model.csc(),
      model.objective(),
      std::move(new_row_lower),
      std::move(new_row_upper),
      model.col_lower(),
      model.col_upper(),
      model.obj_constant());
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

PresolveResult apply_singleton_cols_eq_mutable(const LpModel& model, const PresolveOptions& options);

PresolveResult apply_singleton_cols_eq(const LpModel& model, const PresolveOptions& options) {
  return apply_singleton_cols_eq_mutable(model, options);
}

bool is_integral_ratio(double numerator, double denominator, double tolerance) {
  if (std::fabs(denominator) <= tolerance) {
    return false;
  }
  const double ratio = std::fabs(numerator / denominator);
  return std::isfinite(ratio) && std::fabs(ratio - std::round(ratio)) <= tolerance;
}

bool acceptable_doubleton_pivot(double keep_value, double elim_value) {
  constexpr double kMaxRatio = 1.0e3;
  if (std::fabs(elim_value) <= 0.0) {
    return false;
  }
  const double pivot_ratio = std::fabs(keep_value / elim_value);
  return pivot_ratio <= kMaxRatio && pivot_ratio >= 1.0 / kMaxRatio;
}

void choose_doubleton_columns(int col1,
                              double val1,
                              int col2,
                              double val2,
                              const std::vector<int>& col_nnz,
                              double tolerance,
                              int& elim_col,
                              double& elim_val,
                              int& keep_col,
                              double& keep_val) {
  const bool integral12 = is_integral_ratio(val1, val2, tolerance);
  const bool integral21 = is_integral_ratio(val2, val1, tolerance);

  if (col_nnz[static_cast<std::size_t>(col1)] == 1 &&
      col_nnz[static_cast<std::size_t>(col2)] != 1) {
    elim_col = col1;
    elim_val = val1;
    keep_col = col2;
    keep_val = val2;
  } else if (col_nnz[static_cast<std::size_t>(col1)] != 1 &&
             col_nnz[static_cast<std::size_t>(col2)] == 1) {
    elim_col = col2;
    elim_val = val2;
    keep_col = col1;
    keep_val = val1;
  } else if (integral12 && !integral21) {
    elim_col = col2;
    elim_val = val2;
    keep_col = col1;
    keep_val = val1;
  } else if (integral21 && !integral12) {
    elim_col = col1;
    elim_val = val1;
    keep_col = col2;
    keep_val = val2;
  } else if (col_nnz[static_cast<std::size_t>(col1)] <
             col_nnz[static_cast<std::size_t>(col2)]) {
    elim_col = col1;
    elim_val = val1;
    keep_col = col2;
    keep_val = val2;
  } else {
    elim_col = col2;
    elim_val = val2;
    keep_col = col1;
    keep_val = val1;
  }
}

void doubleton_mapped_interval(double rhs,
                               double keep_value,
                               double elim_value,
                               double elim_lower,
                               double elim_upper,
                               double& mapped_lower,
                               double& mapped_upper) {
  const double alpha = -keep_value / elim_value;
  const double beta = rhs / elim_value;
  mapped_lower = -std::numeric_limits<double>::infinity();
  mapped_upper = std::numeric_limits<double>::infinity();

  if (std::isfinite(elim_lower)) {
    const double bound_value = (elim_lower - beta) / alpha;
    if (alpha > 0.0) {
      mapped_lower = bound_value;
    } else {
      mapped_upper = bound_value;
    }
  }

  if (std::isfinite(elim_upper)) {
    const double bound_value = (elim_upper - beta) / alpha;
    if (alpha > 0.0) {
      mapped_upper = bound_value;
    } else {
      mapped_lower = bound_value;
    }
  }

  if (mapped_lower > mapped_upper) {
    std::swap(mapped_lower, mapped_upper);
  }
}

struct MutableEntry {
  int index = -1;
  double value = 0.0;
};

struct MutableEntryRange {
  MutableEntry* first = nullptr;
  MutableEntry* last = nullptr;

  MutableEntry* begin() const { return first; }
  MutableEntry* end() const { return last; }
  bool empty() const { return first == last; }
  std::size_t size() const { return static_cast<std::size_t>(last - first); }
  MutableEntry& operator[](std::size_t offset) const { return first[offset]; }
  MutableEntry& front() const { return *first; }
  MutableEntry& back() const { return *(last - 1); }
};

struct MutableConstEntryRange {
  const MutableEntry* first = nullptr;
  const MutableEntry* last = nullptr;

  MutableConstEntryRange() = default;
  MutableConstEntryRange(const MutableEntry* first_entry, const MutableEntry* last_entry)
      : first(first_entry), last(last_entry) {}
  MutableConstEntryRange(MutableEntryRange range)
      : first(range.begin()), last(range.end()) {}

  const MutableEntry* begin() const { return first; }
  const MutableEntry* end() const { return last; }
  bool empty() const { return first == last; }
  std::size_t size() const { return static_cast<std::size_t>(last - first); }
  const MutableEntry& operator[](std::size_t offset) const { return first[offset]; }
  const MutableEntry& front() const { return *first; }
  const MutableEntry& back() const { return *(last - 1); }
};

struct MutableRange {
  int start = 0;
  int end = 0;
  int capacity_end = 0;
};

struct DtonRowUpdate {
  bool removed_elim = false;
  bool keep_existed = false;
  bool keep_present_after = false;
  double elim_coeff = 0.0;
  double old_keep_coeff = 0.0;
  double new_keep_coeff = 0.0;
  int new_size = 0;
};

struct MutableColumnFix {
  int col = -1;
  double value = 0.0;
};

class MutablePackedMatrix {
public:
  MutablePackedMatrix() = default;

  MutablePackedMatrix(int major_count, const std::vector<int>& ptr) {
    reset(major_count, ptr);
  }

  void reset(int major_count, const std::vector<int>& ptr) {
    ranges_.assign(static_cast<std::size_t>(major_count + 1), MutableRange{});
    std::size_t total_capacity = 0;
    for (int major = 0; major < major_count; ++major) {
      const int length = ptr[static_cast<std::size_t>(major + 1)] -
                         ptr[static_cast<std::size_t>(major)];
      total_capacity += static_cast<std::size_t>(initial_capacity(length));
    }
    entries_.assign(total_capacity, MutableEntry{});

    int cursor = 0;
    for (int major = 0; major < major_count; ++major) {
      const int length = ptr[static_cast<std::size_t>(major + 1)] -
                         ptr[static_cast<std::size_t>(major)];
      MutableRange& range = ranges_[static_cast<std::size_t>(major)];
      range.start = cursor;
      range.end = cursor;
      range.capacity_end = cursor + initial_capacity(length);
      cursor = range.capacity_end;
    }
    ranges_[static_cast<std::size_t>(major_count)].start = cursor;
    ranges_[static_cast<std::size_t>(major_count)].end = cursor;
    ranges_[static_cast<std::size_t>(major_count)].capacity_end = cursor;
  }

  MutableEntryRange entries(int major) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    return MutableEntryRange{entries_.data() + range.start, entries_.data() + range.end};
  }

  MutableConstEntryRange entries(int major) const {
    const MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    return MutableConstEntryRange{entries_.data() + range.start, entries_.data() + range.end};
  }

  int size(int major) const {
    const MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    return range.end - range.start;
  }

  bool empty(int major) const {
    return size(major) == 0;
  }

  MutableEntry front(int major) const {
    const MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    return entries_[static_cast<std::size_t>(range.start)];
  }

  MutableEntry back(int major) const {
    const MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    return entries_[static_cast<std::size_t>(range.end - 1)];
  }

  void push_back(int major, MutableEntry entry) {
    ensure_extra_capacity(major, 1);
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    entries_[static_cast<std::size_t>(range.end)] = entry;
    ++range.end;
  }

  void pop_back(int major) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    if (range.end > range.start) {
      --range.end;
    }
  }

  void clear(int major) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    range.end = range.start;
  }

  int remove_marked_indices(int major, const std::vector<unsigned char>& removed_index) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    MutableEntry* first = entries_.data() + range.start;
    const int old_length = range.end - range.start;
    int write = 0;
    for (int read = 0; read < old_length; ++read) {
      const int index = first[read].index;
      if (index >= 0 &&
          index < static_cast<int>(removed_index.size()) &&
          removed_index[static_cast<std::size_t>(index)] != 0) {
        continue;
      }
      first[write++] = first[read];
    }
    range.end = range.start + write;
    return write;
  }

  MutableEntry* lower_bound(int major, int index) {
    MutableEntryRange range = entries(major);
    return std::lower_bound(
        range.begin(),
        range.end(),
        index,
        [](const MutableEntry& entry, int target) { return entry.index < target; });
  }

  const MutableEntry* lower_bound(int major, int index) const {
    MutableConstEntryRange range = entries(major);
    return std::lower_bound(
        range.begin(),
        range.end(),
        index,
        [](const MutableEntry& entry, int target) { return entry.index < target; });
  }

  double remove_entry(int major, int index, bool* removed) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    MutableEntry* first = entries_.data() + range.start;
    MutableEntry* last = entries_.data() + range.end;
    MutableEntry* it = std::lower_bound(
        first,
        last,
        index,
        [](const MutableEntry& entry, int target) { return entry.index < target; });
    if (it == last || it->index != index) {
      if (removed != nullptr) {
        *removed = false;
      }
      return 0.0;
    }

    const double old_value = it->value;
    std::move(it + 1, last, it);
    --range.end;
    if (removed != nullptr) {
      *removed = true;
    }
    return old_value;
  }

  void set_entry_value(int major, int index, double value) {
    MutableEntry* it = lower_bound(major, index);
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    if (it != entries_.data() + range.end && it->index == index) {
      it->value = value;
    }
  }

  void insert_entry(int major, int index, double value) {
    ensure_extra_capacity(major, 1);
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    MutableEntry* first = entries_.data() + range.start;
    MutableEntry* last = entries_.data() + range.end;
    MutableEntry* it = std::lower_bound(
        first,
        last,
        index,
        [](const MutableEntry& entry, int target) { return entry.index < target; });
    if (it != last && it->index == index) {
      it->value = value;
      return;
    }
    std::move_backward(it, last, last + 1);
    *it = MutableEntry{index, value};
    ++range.end;
  }

  DtonRowUpdate substitute_doubleton_row_entry(int major,
                                               int elim_index,
                                               int keep_index,
                                               double keep_delta,
                                               double zero_tolerance) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    MutableEntry* first = entries_.data() + range.start;
    const int old_length = range.end - range.start;

    int elim_offset = -1;
    int keep_offset = -1;
    for (int offset = 0; offset < old_length; ++offset) {
      const int index = first[offset].index;
      if (index == elim_index) {
        elim_offset = offset;
      }
      if (index == keep_index) {
        keep_offset = offset;
      }
    }

    DtonRowUpdate update;
    update.new_size = old_length;
    if (elim_offset < 0) {
      return update;
    }

    update.removed_elim = true;
    update.elim_coeff = first[elim_offset].value;
    update.keep_existed = keep_offset >= 0;
    update.old_keep_coeff = update.keep_existed ? first[keep_offset].value : 0.0;
    update.new_keep_coeff = update.old_keep_coeff + keep_delta;
    update.keep_present_after = std::fabs(update.new_keep_coeff) > zero_tolerance;

    if (!update.keep_existed) {
      int write = 0;
      for (int read = 0; read < old_length; ++read) {
        if (read == elim_offset) {
          continue;
        }
        first[write++] = first[read];
      }
      range.end = range.start + write;
      if (update.keep_present_after) {
        insert_entry(major, keep_index, update.new_keep_coeff);
      }
      update.new_size = size(major);
      return update;
    }

    int write = 0;
    for (int read = 0; read < old_length; ++read) {
      if (read == elim_offset) {
        continue;
      }
      if (read == keep_offset) {
        if (update.keep_present_after) {
          first[write++] = MutableEntry{keep_index, update.new_keep_coeff};
        }
        continue;
      }
      first[write++] = first[read];
    }

    range.end = range.start + write;
    update.new_size = write;
    return update;
  }

private:
  static int initial_capacity(int length) {
    return std::max(length + 4, length * 2 + 1);
  }

  void ensure_extra_capacity(int major, int extra) {
    MutableRange& range = ranges_[static_cast<std::size_t>(major)];
    if (range.end + extra <= range.capacity_end) {
      return;
    }

    const int old_length = range.end - range.start;
    const int new_capacity = std::max(old_length + extra + 4, (old_length + extra) * 2 + 1);
    const int new_start = static_cast<int>(entries_.size());
    entries_.resize(entries_.size() + static_cast<std::size_t>(new_capacity));
    std::copy_n(
        entries_.begin() + range.start,
        old_length,
        entries_.begin() + new_start);
    range.start = new_start;
    range.end = new_start + old_length;
    range.capacity_end = new_start + new_capacity;
  }

  std::vector<MutableEntry> entries_;
  std::vector<MutableRange> ranges_;
};

class IntQueue {
public:
  bool empty() const {
    return head_ >= data_.size();
  }

  int front() const {
    return data_[head_];
  }

  void pop_front() {
    ++head_;
    if (head_ == data_.size()) {
      clear();
    }
  }

  void push_back(int value) {
    data_.push_back(value);
  }

  void reserve(std::size_t capacity) {
    data_.reserve(capacity);
  }

  std::size_t remaining_size() const {
    return data_.size() - head_;
  }

  void clear() {
    data_.clear();
    head_ = 0;
  }

private:
  std::vector<int> data_;
  std::size_t head_ = 0;
};

std::uint64_t hash_entry_index_sequence(MutableConstEntryRange entries) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const MutableEntry& entry : entries) {
    hash = mix_hash(hash, static_cast<std::uint64_t>(entry.index));
  }
  return hash;
}

std::uint64_t hash_scaled_entry_value_sequence(MutableConstEntryRange entries) {
  double scale = 0.0;
  double sign = 1.0;
  bool sign_set = false;
  for (const MutableEntry& entry : entries) {
    scale = std::max(scale, std::fabs(entry.value));
    if (!sign_set && std::fabs(entry.value) > 0.0) {
      sign = entry.value < 0.0 ? -1.0 : 1.0;
      sign_set = true;
    }
  }
  if (scale == 0.0) {
    return 0;
  }

  std::uint64_t hash = 1469598103934665603ULL;
  for (const MutableEntry& entry : entries) {
    const double normalized = sign * entry.value / scale;
    const auto quantized = static_cast<std::int64_t>(std::llround(normalized * 1.0e9));
    hash = mix_hash(hash, static_cast<std::uint64_t>(quantized));
  }
  return hash;
}

std::uint64_t hash_julia_parallel_col_entries(MutableConstEntryRange entries) {
  if (entries.empty()) {
    return julia_parallel_hash_mix(0xcbf29ce484222325ULL, 0);
  }
  const double pivot = entries[0].value;
  if (std::fabs(pivot) <= 0.0) {
    return julia_parallel_hash_mix(0xcbf29ce484222325ULL, 0);
  }

  std::uint64_t hash = julia_parallel_hash_mix(
      0xcbf29ce484222325ULL,
      static_cast<std::uint64_t>(entries.size()));
  for (const MutableEntry& entry : entries) {
    hash = julia_parallel_hash_mix(hash, static_cast<std::uint64_t>(entry.index));
    hash = julia_parallel_hash_mix(hash, double_bits(entry.value / pivot));
  }
  return hash;
}

struct MutablePresolveState {
  int num_rows = 0;
  int num_cols = 0;
  double zero_tolerance = 0.0;
  MutablePackedMatrix rows;
  MutablePackedMatrix cols;
  std::vector<int> row_size;
  std::vector<int> col_size;
  std::vector<unsigned char> active_row;
  std::vector<unsigned char> active_col;
  std::vector<unsigned char> empty_row_queued;
  std::vector<unsigned char> singleton_row_queued;
  std::vector<unsigned char> close_bound_col_queued;
  std::vector<unsigned char> empty_col_queued;
  std::vector<unsigned char> dual_fix_col_queued;
  std::vector<unsigned char> singleton_dual_col_queued;
  std::vector<unsigned char> singleton_col_queued;
  std::vector<unsigned char> doubleton_row_queued;
  std::vector<unsigned char> dirty_row_state;
  std::vector<unsigned char> updated_activity_row_queued;
  std::vector<unsigned char> activity_row_dirty;
  std::vector<int> row_work_stamp;
  IntQueue empty_row_queue;
  IntQueue singleton_row_queue;
  IntQueue close_bound_col_queue;
  IntQueue empty_col_queue;
  IntQueue dual_fix_col_queue;
  IntQueue singleton_dual_col_queue;
  IntQueue singleton_col_queue;
  IntQueue doubleton_row_queue;
  IntQueue updated_activity_row_queue;
  IntQueue dirty_row_queue;
  IntQueue next_dirty_row_queue;
  bool empty_rows_seeded = false;
  bool singleton_rows_seeded = false;
  bool close_bound_cols_seeded = false;
  bool empty_cols_seeded = false;
  bool dual_fix_cols_seeded = false;
  bool singleton_dual_cols_seeded = false;
  bool singleton_cols_seeded = false;
  bool doubleton_rows_seeded = false;
  bool activity_checks_seeded = false;
  bool propagation_rows_seeded = false;
  bool propagation_active = false;
  bool activity_cache_active = false;
  std::vector<double> objective;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  double obj_constant = 0.0;
  double objective_delta = 0.0;
  int reduced_rows = 0;
  int active_nonzeros = 0;
  int current_row_work_stamp = 0;
  int doubleton_eq_reductions_total = 0;
  std::vector<CachedRowActivity> row_activity;
  int structural_change_version = 0;
  int cached_row_parallel_signature_version = -1;
  int cached_col_parallel_signature_version = -1;
  std::vector<SparsePatternSignature> cached_row_signatures;
  std::vector<SparsePatternSignature> cached_col_signatures;

  void touch_structure_changed() {
    ++structural_change_version;
    if (structural_change_version == std::numeric_limits<int>::max()) {
      structural_change_version = 1;
    }
  }

  void touch_row_parallel_signatures() {
    cached_row_parallel_signature_version = -1;
  }

  void touch_col_parallel_signatures() {
    cached_col_parallel_signature_version = -1;
  }

  void touch_parallel_signatures() {
    cached_row_parallel_signature_version = -1;
    cached_col_parallel_signature_version = -1;
  }

  const std::vector<SparsePatternSignature>& row_pattern_signatures() {
    if (cached_row_parallel_signature_version == structural_change_version) {
      return cached_row_signatures;
    }
    cached_row_signatures.clear();
    cached_row_signatures.reserve(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
      if (active_row[static_cast<std::size_t>(row)] == 0 ||
          row_size[static_cast<std::size_t>(row)] <= 0) {
        continue;
      }
      const MutableConstEntryRange entries = rows.entries(row);
      cached_row_signatures.push_back(SparsePatternSignature{
          row,
          row_size[static_cast<std::size_t>(row)],
          hash_entry_index_sequence(entries),
          hash_scaled_entry_value_sequence(entries)});
    }
    std::sort(
        cached_row_signatures.begin(),
        cached_row_signatures.end(),
        signature_less);
    cached_row_parallel_signature_version = structural_change_version;
    return cached_row_signatures;
  }

  const std::vector<SparsePatternSignature>& col_pattern_signatures() {
    if (cached_col_parallel_signature_version == structural_change_version) {
      return cached_col_signatures;
    }
    cached_col_signatures.clear();
    cached_col_signatures.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      if (active_col[static_cast<std::size_t>(col)] == 0 ||
          col_size[static_cast<std::size_t>(col)] <= 0) {
        continue;
      }
      const MutableConstEntryRange entries = cols.entries(col);
      cached_col_signatures.push_back(SparsePatternSignature{
          col,
          col_size[static_cast<std::size_t>(col)],
          hash_julia_parallel_col_entries(entries),
          0});
    }
    std::sort(
        cached_col_signatures.begin(),
        cached_col_signatures.end(),
        signature_less);
    cached_col_parallel_signature_version = structural_change_version;
    return cached_col_signatures;
  }

  MutablePresolveState(const LpModel& model, double tolerance, double bound_tolerance)
      : num_rows(model.num_rows()),
        num_cols(model.num_cols()),
        zero_tolerance(tolerance),
        rows(num_rows, model.csr().row_ptr()),
        cols(num_cols, model.csc().col_ptr()),
        row_size(static_cast<std::size_t>(num_rows), 0),
        col_size(static_cast<std::size_t>(num_cols), 0),
        active_row(static_cast<std::size_t>(num_rows), 1),
        active_col(static_cast<std::size_t>(num_cols), 1),
        empty_row_queued(static_cast<std::size_t>(num_rows), 0),
        singleton_row_queued(static_cast<std::size_t>(num_rows), 0),
        close_bound_col_queued(static_cast<std::size_t>(num_cols), 0),
        empty_col_queued(static_cast<std::size_t>(num_cols), 0),
        dual_fix_col_queued(static_cast<std::size_t>(num_cols), 0),
        singleton_dual_col_queued(static_cast<std::size_t>(num_cols), 0),
        singleton_col_queued(static_cast<std::size_t>(num_cols), 0),
        doubleton_row_queued(static_cast<std::size_t>(num_rows), 0),
        dirty_row_state(static_cast<std::size_t>(num_rows), 0),
        updated_activity_row_queued(static_cast<std::size_t>(num_rows), 0),
        activity_row_dirty(static_cast<std::size_t>(num_rows), 1),
        row_work_stamp(static_cast<std::size_t>(num_rows), 0),
        objective(model.objective()),
        row_lower(model.row_lower()),
        row_upper(model.row_upper()),
        col_lower(model.col_lower()),
        col_upper(model.col_upper()),
        obj_constant(model.obj_constant()) {
    row_activity.resize(static_cast<std::size_t>(num_rows));
    const CsrMatrix& csr = model.csr();
    for (int row = 0; row < num_rows; ++row) {
      for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
        const double value = csr.values()[pos];
        if (std::fabs(value) <= zero_tolerance) {
          continue;
        }
        rows.push_back(row, MutableEntry{csr.col_idx()[pos], value});
      }
      row_size[static_cast<std::size_t>(row)] = rows.size(row);
      active_nonzeros += row_size[static_cast<std::size_t>(row)];
    }

    const CscMatrix& csc = model.csc();
    for (int col = 0; col < num_cols; ++col) {
      for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
        const double value = csc.values()[pos];
        if (std::fabs(value) <= zero_tolerance) {
          continue;
        }
        cols.push_back(col, MutableEntry{csc.row_idx()[pos], value});
      }
      col_size[static_cast<std::size_t>(col)] = cols.size(col);
    }

    empty_row_queue.reserve(static_cast<std::size_t>(num_rows));
    singleton_row_queue.reserve(static_cast<std::size_t>(num_rows));
    close_bound_col_queue.reserve(static_cast<std::size_t>(num_cols));
    empty_col_queue.reserve(static_cast<std::size_t>(num_cols));
    dual_fix_col_queue.reserve(static_cast<std::size_t>(num_cols));
    singleton_dual_col_queue.reserve(static_cast<std::size_t>(num_cols));
    singleton_col_queue.reserve(static_cast<std::size_t>(num_cols));
    doubleton_row_queue.reserve(static_cast<std::size_t>(num_rows));
    updated_activity_row_queue.reserve(static_cast<std::size_t>(num_rows));
    dirty_row_queue.reserve(static_cast<std::size_t>(num_rows));
    next_dirty_row_queue.reserve(static_cast<std::size_t>(num_rows / 16 + 1));

    activity_checks_seeded = true;
    propagation_rows_seeded = true;
    empty_rows_seeded = true;
    singleton_rows_seeded = true;
    close_bound_cols_seeded = true;
    empty_cols_seeded = true;
    dual_fix_cols_seeded = true;
    singleton_dual_cols_seeded = true;
    singleton_cols_seeded = true;
    doubleton_rows_seeded = true;

    for (int row = 0; row < num_rows; ++row) {
      if (row_size[static_cast<std::size_t>(row)] == 0) {
        enqueue_empty_row(row);
      }
      if (row_size[static_cast<std::size_t>(row)] == 1) {
        enqueue_singleton_row(row);
      }
      if (row_size[static_cast<std::size_t>(row)] == 2 &&
          is_equality_row(row_lower[static_cast<std::size_t>(row)],
                          row_upper[static_cast<std::size_t>(row)],
                          bound_tolerance)) {
        enqueue_doubleton_row(row, bound_tolerance);
      }
      enqueue_updated_activity_row(row, true);
      enqueue_dirty_row(row);
    }

    for (int col = 0; col < num_cols; ++col) {
      if (col_size[static_cast<std::size_t>(col)] == 0) {
        enqueue_empty_col(col);
      }
      if (col_size[static_cast<std::size_t>(col)] == 1) {
        enqueue_singleton_dual_col(col);
        enqueue_singleton_col(col);
      }
      if (is_close_fixed_column(
              col_lower[static_cast<std::size_t>(col)],
              col_upper[static_cast<std::size_t>(col)],
              bound_tolerance)) {
        enqueue_close_bound_col(col);
      }
      enqueue_dual_fix_col(col);
    }

    reduced_rows = num_rows;
  }

  int active_nnz() const {
    return active_nonzeros;
  }

  int next_row_work_stamp() {
    ++current_row_work_stamp;
    if (current_row_work_stamp == std::numeric_limits<int>::max()) {
      std::fill(row_work_stamp.begin(), row_work_stamp.end(), 0);
      current_row_work_stamp = 1;
    }
    return current_row_work_stamp;
  }

  bool mark_row_once(int row, int stamp) {
    if (row < 0 || row >= num_rows) {
      return false;
    }
    int& current = row_work_stamp[static_cast<std::size_t>(row)];
    if (current == stamp) {
      return false;
    }
    current = stamp;
    return true;
  }

  void enqueue_updated_activity_row(int row, bool force = false) {
    if ((!force && !activity_checks_seeded) ||
        row < 0 || row >= num_rows ||
        active_row[static_cast<std::size_t>(row)] == 0 ||
        updated_activity_row_queued[static_cast<std::size_t>(row)] != 0) {
      return;
    }
    updated_activity_row_queue.push_back(row);
    updated_activity_row_queued[static_cast<std::size_t>(row)] = 1;
  }

  void seed_updated_activity_rows() {
    if (activity_checks_seeded) {
      return;
    }
    activity_checks_seeded = true;
    updated_activity_row_queue.reserve(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
      enqueue_updated_activity_row(row, true);
    }
  }

  void mark_row_activity_dirty(int row) {
    if (!activity_cache_active && !activity_checks_seeded) {
      return;
    }
    if (row >= 0 && row < num_rows) {
      enqueue_updated_activity_row(row);
      activity_row_dirty[static_cast<std::size_t>(row)] = 1;
    }
  }

  void mark_rows_for_col_activity_dirty(int col) {
    if (!activity_cache_active) {
      return;
    }
    if (col < 0 || col >= num_cols) {
      return;
    }
    for (const MutableEntry& entry : cols.entries(col)) {
      if (active_row[static_cast<std::size_t>(entry.index)] != 0) {
        mark_row_activity_dirty(entry.index);
      }
    }
  }

  const CachedRowActivity& ensure_row_activity(int row) {
    activity_cache_active = true;
    if (activity_row_dirty[static_cast<std::size_t>(row)] == 0) {
      return row_activity[static_cast<std::size_t>(row)];
    }

    CachedRowActivity activity;
    if (active_row[static_cast<std::size_t>(row)] != 0) {
      for (const MutableEntry& entry : rows.entries(row)) {
        const int col = entry.index;
        if (active_col[static_cast<std::size_t>(col)] == 0) {
          continue;
        }
        add_activity_value(
            activity.minimum,
            term_min(entry.value,
                     col_lower[static_cast<std::size_t>(col)],
                     col_upper[static_cast<std::size_t>(col)]));
        add_activity_value(
            activity.maximum,
            term_max(entry.value,
                     col_lower[static_cast<std::size_t>(col)],
                     col_upper[static_cast<std::size_t>(col)]));
      }
    }
    row_activity[static_cast<std::size_t>(row)] = activity;
    activity_row_dirty[static_cast<std::size_t>(row)] = 0;
    return row_activity[static_cast<std::size_t>(row)];
  }

  bool row_is_active_doubleton_eq(int row, double bound_tolerance) const {
    return row >= 0 && row < num_rows &&
           active_row[static_cast<std::size_t>(row)] != 0 &&
           row_size[static_cast<std::size_t>(row)] == 2 &&
           is_equality_row(row_lower[static_cast<std::size_t>(row)],
                           row_upper[static_cast<std::size_t>(row)],
                           bound_tolerance);
  }

  bool col_is_active_singleton(int col) const {
    return col >= 0 && col < num_cols &&
           active_col[static_cast<std::size_t>(col)] != 0 &&
           col_size[static_cast<std::size_t>(col)] == 1;
  }

  bool row_is_active_empty(int row) const {
    return row >= 0 && row < num_rows &&
           active_row[static_cast<std::size_t>(row)] != 0 &&
           row_size[static_cast<std::size_t>(row)] == 0;
  }

  bool row_is_active_singleton(int row) const {
    return row >= 0 && row < num_rows &&
           active_row[static_cast<std::size_t>(row)] != 0 &&
           row_size[static_cast<std::size_t>(row)] == 1;
  }

  bool col_is_active_empty(int col) const {
    return col >= 0 && col < num_cols &&
           active_col[static_cast<std::size_t>(col)] != 0 &&
           col_size[static_cast<std::size_t>(col)] == 0;
  }

  void enqueue_empty_row(int row) {
    if (!row_is_active_empty(row) ||
        empty_row_queued[static_cast<std::size_t>(row)] != 0) {
      return;
    }
    empty_row_queue.push_back(row);
    empty_row_queued[static_cast<std::size_t>(row)] = 1;
  }

  void seed_empty_rows() {
    if (empty_rows_seeded) {
      return;
    }
    empty_row_queue.reserve(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
      enqueue_empty_row(row);
    }
    empty_rows_seeded = true;
  }

  void enqueue_singleton_row(int row) {
    if (!row_is_active_singleton(row) ||
        singleton_row_queued[static_cast<std::size_t>(row)] != 0) {
      return;
    }
    singleton_row_queue.push_back(row);
    singleton_row_queued[static_cast<std::size_t>(row)] = 1;
  }

  void seed_singleton_rows() {
    if (singleton_rows_seeded) {
      return;
    }
    singleton_row_queue.reserve(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
      enqueue_singleton_row(row);
    }
    singleton_rows_seeded = true;
  }

  void enqueue_close_bound_col(int col) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0 ||
        close_bound_col_queued[static_cast<std::size_t>(col)] != 0) {
      return;
    }
    close_bound_col_queue.push_back(col);
    close_bound_col_queued[static_cast<std::size_t>(col)] = 1;
  }

  void seed_close_bound_cols() {
    if (close_bound_cols_seeded) {
      return;
    }
    close_bound_col_queue.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      enqueue_close_bound_col(col);
    }
    close_bound_cols_seeded = true;
  }

  void enqueue_empty_col(int col) {
    if (!col_is_active_empty(col) ||
        empty_col_queued[static_cast<std::size_t>(col)] != 0) {
      return;
    }
    empty_col_queue.push_back(col);
    empty_col_queued[static_cast<std::size_t>(col)] = 1;
  }

  void seed_empty_cols() {
    if (empty_cols_seeded) {
      return;
    }
    empty_col_queue.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      enqueue_empty_col(col);
    }
    empty_cols_seeded = true;
  }

  void enqueue_dual_fix_col(int col) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0 ||
        dual_fix_col_queued[static_cast<std::size_t>(col)] != 0) {
      return;
    }
    dual_fix_col_queue.push_back(col);
    dual_fix_col_queued[static_cast<std::size_t>(col)] = 1;
  }

  void seed_dual_fix_cols() {
    if (dual_fix_cols_seeded) {
      return;
    }
    dual_fix_col_queue.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      enqueue_dual_fix_col(col);
    }
    dual_fix_cols_seeded = true;
  }

  void enqueue_singleton_dual_col(int col) {
    if (!col_is_active_singleton(col) ||
        singleton_dual_col_queued[static_cast<std::size_t>(col)] != 0) {
      return;
    }
    singleton_dual_col_queue.push_back(col);
    singleton_dual_col_queued[static_cast<std::size_t>(col)] = 1;
  }

  void seed_singleton_dual_cols() {
    if (singleton_dual_cols_seeded) {
      return;
    }
    singleton_dual_col_queue.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      enqueue_singleton_dual_col(col);
    }
    singleton_dual_cols_seeded = true;
  }

  void enqueue_singleton_col(int col) {
    if (!col_is_active_singleton(col)) {
      return;
    }
    if (singleton_col_queued[static_cast<std::size_t>(col)] == 0) {
      singleton_col_queue.push_back(col);
      singleton_col_queued[static_cast<std::size_t>(col)] = 1;
    }
  }

  void seed_singleton_cols() {
    if (singleton_cols_seeded) {
      return;
    }
    singleton_col_queue.reserve(static_cast<std::size_t>(num_cols));
    for (int col = 0; col < num_cols; ++col) {
      enqueue_singleton_col(col);
    }
    singleton_cols_seeded = true;
  }

  void enqueue_doubleton_row(int row, double bound_tolerance) {
    if (!row_is_active_doubleton_eq(row, bound_tolerance) ||
        doubleton_row_queued[static_cast<std::size_t>(row)] != 0) {
      return;
    }
    doubleton_row_queue.push_back(row);
    doubleton_row_queued[static_cast<std::size_t>(row)] = 1;
  }

  void seed_doubleton_rows(double bound_tolerance) {
    if (doubleton_rows_seeded) {
      return;
    }
    doubleton_row_queue.reserve(static_cast<std::size_t>(num_rows));
    for (int row = 0; row < num_rows; ++row) {
      enqueue_doubleton_row(row, bound_tolerance);
    }
    doubleton_rows_seeded = true;
  }

  void enqueue_dirty_row(int row) {
    if (row < 0 || row >= num_rows ||
        active_row[static_cast<std::size_t>(row)] == 0) {
      return;
    }
    unsigned char& state = dirty_row_state[static_cast<std::size_t>(row)];
    if (state == 1 || state == 3) {
      return;
    }
    if (propagation_active && state == 2) {
      next_dirty_row_queue.push_back(row);
      state = 3;
      return;
    }
    if (state != 0) {
      return;
    }
    dirty_row_queue.push_back(row);
    state = 1;
  }

  void seed_propagation_rows() {
    if (propagation_rows_seeded) {
      return;
    }
    dirty_row_queue.reserve(static_cast<std::size_t>(num_rows));
    next_dirty_row_queue.reserve(static_cast<std::size_t>(num_rows / 16 + 1));
    for (int row = 0; row < num_rows; ++row) {
      enqueue_dirty_row(row);
    }
    propagation_rows_seeded = true;
  }

  void enqueue_rows_for_col(int col) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0) {
      return;
    }
    for (const MutableEntry& entry : cols.entries(col)) {
      mark_row_activity_dirty(entry.index);
      enqueue_dirty_row(entry.index);
    }
  }

  void note_col_bounds_changed(int col) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0) {
      return;
    }
    enqueue_close_bound_col(col);
    enqueue_dual_fix_col(col);
    enqueue_rows_for_col(col);
  }

  bool tighten_col_lower(int col, double lower) {
    double& current = col_lower[static_cast<std::size_t>(col)];
    if (lower <= current) {
      return false;
    }
    current = lower;
    note_col_bounds_changed(col);
    return true;
  }

  bool tighten_col_upper(int col, double upper) {
    double& current = col_upper[static_cast<std::size_t>(col)];
    if (upper >= current) {
      return false;
    }
    current = upper;
    note_col_bounds_changed(col);
    return true;
  }

  bool set_tighter_col_bounds(int col, double lower, double upper) {
    bool changed = false;
    if (lower > col_lower[static_cast<std::size_t>(col)]) {
      col_lower[static_cast<std::size_t>(col)] = lower;
      changed = true;
    }
    if (upper < col_upper[static_cast<std::size_t>(col)]) {
      col_upper[static_cast<std::size_t>(col)] = upper;
      changed = true;
    }
    if (changed) {
      note_col_bounds_changed(col);
    }
    return changed;
  }

  void set_col_bounds(int col, double lower, double upper) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0) {
      return;
    }
    bool changed = false;
    double& current_lower = col_lower[static_cast<std::size_t>(col)];
    double& current_upper = col_upper[static_cast<std::size_t>(col)];
    if (current_lower != lower || current_upper != upper) {
      current_lower = lower;
      current_upper = upper;
      changed = true;
    }
    if (changed) {
      note_col_bounds_changed(col);
    }
  }

  void note_row_bounds_changed(int row, double bound_tolerance) {
    if (row < 0 || row >= num_rows ||
        active_row[static_cast<std::size_t>(row)] == 0) {
      return;
    }
    mark_row_activity_dirty(row);
    enqueue_dirty_row(row);
    enqueue_empty_row(row);
    enqueue_singleton_row(row);
    enqueue_doubleton_row(row, bound_tolerance);
    for (const MutableEntry& entry : rows.entries(row)) {
      enqueue_dual_fix_col(entry.index);
    }
  }

  bool set_row_bounds(int row, double lower, double upper, double bound_tolerance) {
    double& current_lower = row_lower[static_cast<std::size_t>(row)];
    double& current_upper = row_upper[static_cast<std::size_t>(row)];
    if (lower == current_lower && upper == current_upper) {
      return false;
    }
    current_lower = lower;
    current_upper = upper;
    note_row_bounds_changed(row, bound_tolerance);
    return true;
  }

  bool shift_row_bounds(int row, double shift, double bound_tolerance) {
    if (shift == 0.0) {
      return false;
    }
    row_lower[static_cast<std::size_t>(row)] -= shift;
    row_upper[static_cast<std::size_t>(row)] -= shift;
    note_row_bounds_changed(row, bound_tolerance);
    return true;
  }

  bool shift_row_bounds_preserving_locks(int row, double shift, double bound_tolerance) {
    if (shift == 0.0) {
      return false;
    }
    row_lower[static_cast<std::size_t>(row)] -= shift;
    row_upper[static_cast<std::size_t>(row)] -= shift;
    mark_row_activity_dirty(row);
    enqueue_dirty_row(row);
    enqueue_doubleton_row(row, bound_tolerance);
    return true;
  }

  void set_col_entry_value(int col, int row, double value) {
    cols.set_entry_value(col, row, value);
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
  }

  void insert_col_entry(int col, int row, double value) {
    cols.insert_entry(col, row, value);
    ++col_size[static_cast<std::size_t>(col)];
    touch_structure_changed();
    touch_parallel_signatures();
  }

  void remove_col_entry(int col, int row) {
    bool removed = false;
    cols.remove_entry(col, row, &removed);
    if (removed) {
      --col_size[static_cast<std::size_t>(col)];
      touch_structure_changed();
      touch_parallel_signatures();
    }
  }

  void add_or_update_row_entry(int row, int col, double delta) {
    if (active_row[static_cast<std::size_t>(row)] == 0 ||
        active_col[static_cast<std::size_t>(col)] == 0 ||
        std::fabs(delta) <= zero_tolerance) {
      return;
    }

    MutableEntry* it = rows.lower_bound(row, col);
    MutableEntryRange row_entries = rows.entries(row);
    if (it == row_entries.end() || it->index != col) {
      rows.insert_entry(row, col, delta);
      ++row_size[static_cast<std::size_t>(row)];
      ++active_nonzeros;
      insert_col_entry(col, row, delta);
      touch_structure_changed();
      touch_row_parallel_signatures();
      touch_col_parallel_signatures();
      mark_row_activity_dirty(row);
      return;
    }

    const double new_value = it->value + delta;
    if (std::fabs(new_value) <= zero_tolerance) {
      bool removed = false;
      rows.remove_entry(row, col, &removed);
      --row_size[static_cast<std::size_t>(row)];
      --active_nonzeros;
      remove_col_entry(col, row);
      enqueue_empty_row(row);
      enqueue_singleton_row(row);
      enqueue_empty_col(col);
      enqueue_singleton_dual_col(col);
      enqueue_singleton_col(col);
      enqueue_dual_fix_col(col);
      touch_structure_changed();
      touch_row_parallel_signatures();
      touch_col_parallel_signatures();
      mark_row_activity_dirty(row);
    } else {
      it->value = new_value;
      set_col_entry_value(col, row, new_value);
      touch_structure_changed();
      touch_row_parallel_signatures();
      touch_col_parallel_signatures();
      mark_row_activity_dirty(row);
    }
  }

  bool row_activity_bounds_excluding(int row,
                                     int excluded_col,
                                     double& rest_min,
                                     double& rest_max) {
    rest_min = 0.0;
    rest_max = 0.0;
    for (const MutableEntry& entry : rows.entries(row)) {
      const int col = entry.index;
      if (col == excluded_col || active_col[static_cast<std::size_t>(col)] == 0) {
        continue;
      }
      const double min_term = term_min(
          entry.value,
          col_lower[static_cast<std::size_t>(col)],
          col_upper[static_cast<std::size_t>(col)]);
      const double max_term = term_max(
          entry.value,
          col_lower[static_cast<std::size_t>(col)],
          col_upper[static_cast<std::size_t>(col)]);
      if (std::isnan(min_term) || std::isnan(max_term)) {
        return false;
      }
      rest_min += min_term;
      rest_max += max_term;
      if (std::isnan(rest_min) || std::isnan(rest_max)) {
        return false;
      }
    }
    return true;
  }

  bool remove_row_col_entry(int row, int col, double* old_value) {
    bool removed = false;
    const double value = rows.remove_entry(row, col, &removed);
    if (!removed) {
      if (old_value != nullptr) {
        *old_value = 0.0;
      }
      return false;
    }
    --row_size[static_cast<std::size_t>(row)];
    --active_nonzeros;
    remove_col_entry(col, row);
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
    enqueue_empty_row(row);
    enqueue_singleton_row(row);
    enqueue_empty_col(col);
    enqueue_singleton_dual_col(col);
    enqueue_singleton_col(col);
    enqueue_dual_fix_col(col);
    mark_row_activity_dirty(row);
    if (old_value != nullptr) {
      *old_value = value;
    }
    return true;
  }

  bool remove_row_entry_only(int row, int col, double* old_value) {
    bool removed = false;
    const double value = rows.remove_entry(row, col, &removed);
    if (!removed) {
      if (old_value != nullptr) {
        *old_value = 0.0;
      }
      return false;
    }
    --row_size[static_cast<std::size_t>(row)];
    --active_nonzeros;
    touch_structure_changed();
    touch_row_parallel_signatures();
    enqueue_empty_row(row);
    enqueue_singleton_row(row);
    mark_row_activity_dirty(row);
    if (old_value != nullptr) {
      *old_value = value;
    }
    return true;
  }

  bool substitute_doubleton_row_entry(int row,
                                      int elim_col,
                                      int keep_col,
                                      double keep_delta,
                                      double* elim_coeff) {
    const int old_size = row_size[static_cast<std::size_t>(row)];
    DtonRowUpdate update =
        rows.substitute_doubleton_row_entry(row, elim_col, keep_col, keep_delta, zero_tolerance);
    if (!update.removed_elim) {
      if (elim_coeff != nullptr) {
        *elim_coeff = 0.0;
      }
      return false;
    }

    row_size[static_cast<std::size_t>(row)] = update.new_size;
    active_nonzeros += update.new_size - old_size;
    if (elim_coeff != nullptr) {
      *elim_coeff = update.elim_coeff;
    }

    if (update.keep_existed) {
      if (update.keep_present_after) {
        set_col_entry_value(keep_col, row, update.new_keep_coeff);
      } else {
        remove_col_entry(keep_col, row);
      }
    } else if (update.keep_present_after) {
      insert_col_entry(keep_col, row, update.new_keep_coeff);
    }

    enqueue_empty_row(row);
    enqueue_singleton_row(row);
    enqueue_empty_col(keep_col);
    enqueue_singleton_dual_col(keep_col);
    enqueue_singleton_col(keep_col);
    enqueue_dual_fix_col(keep_col);
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
    mark_row_activity_dirty(row);
    return true;
  }

  void deactivate_empty_col(int col) {
    active_col[static_cast<std::size_t>(col)] = 0;
    cols.clear(col);
    col_size[static_cast<std::size_t>(col)] = 0;
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
    close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
    empty_col_queued[static_cast<std::size_t>(col)] = 0;
    dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_dual_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_col_queued[static_cast<std::size_t>(col)] = 0;
  }

  void remove_active_col(int col) {
    while (!cols.entries(col).empty()) {
      const MutableEntry first = cols.entries(col).front();
      remove_row_col_entry(first.index, col, nullptr);
    }
    active_col[static_cast<std::size_t>(col)] = 0;
    cols.clear(col);
    col_size[static_cast<std::size_t>(col)] = 0;
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
    close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
    empty_col_queued[static_cast<std::size_t>(col)] = 0;
    dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_dual_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_col_queued[static_cast<std::size_t>(col)] = 0;
  }

  void remove_active_row(int row) {
    std::vector<int> row_cols;
    row_cols.reserve(rows.entries(row).size());
    for (const MutableEntry& entry : rows.entries(row)) {
      row_cols.push_back(entry.index);
    }
    for (int col : row_cols) {
      remove_col_entry(col, row);
      enqueue_empty_col(col);
      enqueue_singleton_dual_col(col);
      enqueue_singleton_col(col);
      enqueue_dual_fix_col(col);
    }
    rows.clear(row);
    active_nonzeros -= row_size[static_cast<std::size_t>(row)];
    row_size[static_cast<std::size_t>(row)] = 0;
    active_row[static_cast<std::size_t>(row)] = 0;
    empty_row_queued[static_cast<std::size_t>(row)] = 0;
    singleton_row_queued[static_cast<std::size_t>(row)] = 0;
    doubleton_row_queued[static_cast<std::size_t>(row)] = 0;
    updated_activity_row_queued[static_cast<std::size_t>(row)] = 0;
    dirty_row_state[static_cast<std::size_t>(row)] = 0;
    mark_row_activity_dirty(row);
    --reduced_rows;
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
  }

  void remove_doubleton_pivot_row(int row, int elim_col, int keep_col) {
    (void)elim_col;
    remove_col_entry(keep_col, row);
    enqueue_empty_col(keep_col);
    enqueue_singleton_dual_col(keep_col);
    enqueue_singleton_col(keep_col);
    enqueue_dual_fix_col(keep_col);

    rows.clear(row);
    active_nonzeros -= row_size[static_cast<std::size_t>(row)];
    row_size[static_cast<std::size_t>(row)] = 0;
    active_row[static_cast<std::size_t>(row)] = 0;
    empty_row_queued[static_cast<std::size_t>(row)] = 0;
    singleton_row_queued[static_cast<std::size_t>(row)] = 0;
    doubleton_row_queued[static_cast<std::size_t>(row)] = 0;
    updated_activity_row_queued[static_cast<std::size_t>(row)] = 0;
    dirty_row_state[static_cast<std::size_t>(row)] = 0;
    mark_row_activity_dirty(row);
    --reduced_rows;
    touch_structure_changed();
    touch_row_parallel_signatures();
    touch_col_parallel_signatures();
  }

  void fix_active_col(int col, double value) {
    std::vector<MutableEntry> col_entries(
        cols.entries(col).begin(),
        cols.entries(col).end());
    for (const MutableEntry& entry : col_entries) {
      const int row = entry.index;
      if (active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      row_lower[static_cast<std::size_t>(row)] -= entry.value * value;
      row_upper[static_cast<std::size_t>(row)] -= entry.value * value;
      remove_row_col_entry(row, col, nullptr);
      enqueue_dirty_row(row);
    }
    objective_delta += objective[static_cast<std::size_t>(col)] * value;
    active_col[static_cast<std::size_t>(col)] = 0;
    cols.clear(col);
    col_size[static_cast<std::size_t>(col)] = 0;
    close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
    empty_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_dual_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_col_queued[static_cast<std::size_t>(col)] = 0;
    dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
    touch_structure_changed();
    touch_parallel_signatures();
  }

  void fix_active_cols_batch(const std::vector<MutableColumnFix>& fixes,
                             double bound_tolerance) {
    if (fixes.empty()) {
      return;
    }

    std::vector<unsigned char> fixed_col(static_cast<std::size_t>(num_cols), 0);
    std::vector<int> touched_rows;
    touched_rows.reserve(fixes.size());
    const int row_stamp = next_row_work_stamp();
    bool changed = false;

    for (const MutableColumnFix& fix : fixes) {
      const int col = fix.col;
      if (col < 0 || col >= num_cols ||
          active_col[static_cast<std::size_t>(col)] == 0 ||
          fixed_col[static_cast<std::size_t>(col)] != 0) {
        continue;
      }

      fixed_col[static_cast<std::size_t>(col)] = 1;
      objective_delta += objective[static_cast<std::size_t>(col)] * fix.value;

      for (const MutableEntry& entry : cols.entries(col)) {
        const int row = entry.index;
        if (active_row[static_cast<std::size_t>(row)] == 0) {
          continue;
        }
        row_lower[static_cast<std::size_t>(row)] -= entry.value * fix.value;
        row_upper[static_cast<std::size_t>(row)] -= entry.value * fix.value;
        if (mark_row_once(row, row_stamp)) {
          touched_rows.push_back(row);
        }
      }

      active_col[static_cast<std::size_t>(col)] = 0;
      cols.clear(col);
      col_size[static_cast<std::size_t>(col)] = 0;
      close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
      empty_col_queued[static_cast<std::size_t>(col)] = 0;
      singleton_dual_col_queued[static_cast<std::size_t>(col)] = 0;
      singleton_col_queued[static_cast<std::size_t>(col)] = 0;
      dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
      changed = true;
    }

    for (int row : touched_rows) {
      if (active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      const int old_size = row_size[static_cast<std::size_t>(row)];
      const int new_size = rows.remove_marked_indices(row, fixed_col);
      if (new_size != old_size) {
        active_nonzeros += new_size - old_size;
        row_size[static_cast<std::size_t>(row)] = new_size;
        enqueue_empty_row(row);
        enqueue_singleton_row(row);
        enqueue_doubleton_row(row, bound_tolerance);
      }
      mark_row_activity_dirty(row);
      enqueue_dirty_row(row);
    }

    if (changed) {
      touch_structure_changed();
      touch_parallel_signatures();
    }
  }

  bool fix_empty_col_without_touch(int col, double value) {
    if (col < 0 || col >= num_cols ||
        active_col[static_cast<std::size_t>(col)] == 0 ||
        col_size[static_cast<std::size_t>(col)] != 0) {
      return false;
    }

    objective_delta += objective[static_cast<std::size_t>(col)] * value;
    active_col[static_cast<std::size_t>(col)] = 0;
    cols.clear(col);
    close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
    empty_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_dual_col_queued[static_cast<std::size_t>(col)] = 0;
    singleton_col_queued[static_cast<std::size_t>(col)] = 0;
    dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
    return true;
  }

  std::vector<int> active_column_map() const {
    std::vector<int> old_to_new_col(static_cast<std::size_t>(num_cols), -1);
    int next_col = 0;
    for (int col = 0; col < num_cols; ++col) {
      if (active_col[static_cast<std::size_t>(col)] == 0) {
        continue;
      }
      old_to_new_col[static_cast<std::size_t>(col)] = next_col++;
    }
    return old_to_new_col;
  }

  LpModel export_model(const std::vector<int>& old_to_new_col) const {
    std::vector<int> old_to_new_row(static_cast<std::size_t>(num_rows), -1);
    std::vector<double> kept_row_lower;
    std::vector<double> kept_row_upper;
    kept_row_lower.reserve(static_cast<std::size_t>(reduced_rows));
    kept_row_upper.reserve(static_cast<std::size_t>(reduced_rows));

    int next_row = 0;
    for (int row = 0; row < num_rows; ++row) {
      if (active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
      kept_row_lower.push_back(row_lower[static_cast<std::size_t>(row)]);
      kept_row_upper.push_back(row_upper[static_cast<std::size_t>(row)]);
    }

    std::vector<double> kept_objective;
    std::vector<double> kept_col_lower;
    std::vector<double> kept_col_upper;
    kept_objective.reserve(static_cast<std::size_t>(num_cols));
    kept_col_lower.reserve(static_cast<std::size_t>(num_cols));
    kept_col_upper.reserve(static_cast<std::size_t>(num_cols));

    int next_col = 0;
    for (int col = 0; col < num_cols; ++col) {
      if (active_col[static_cast<std::size_t>(col)] == 0) {
        continue;
      }
      if (old_to_new_col[static_cast<std::size_t>(col)] != next_col) {
        throw std::logic_error("column export map is inconsistent");
      }
      kept_objective.push_back(objective[static_cast<std::size_t>(col)]);
      kept_col_lower.push_back(col_lower[static_cast<std::size_t>(col)]);
      kept_col_upper.push_back(col_upper[static_cast<std::size_t>(col)]);
      ++next_col;
    }

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    row_ptr.reserve(static_cast<std::size_t>(next_row + 1));
    const int nnz_hint = active_nnz();
    col_idx.reserve(static_cast<std::size_t>(nnz_hint));
    values.reserve(static_cast<std::size_t>(nnz_hint));
    row_ptr.push_back(0);
    for (int row = 0; row < num_rows; ++row) {
      if (active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      for (const MutableEntry& entry : rows.entries(row)) {
        const int new_col = old_to_new_col[static_cast<std::size_t>(entry.index)];
        if (new_col < 0) {
          continue;
        }
        col_idx.push_back(new_col);
        values.push_back(entry.value);
      }
      row_ptr.push_back(static_cast<int>(values.size()));
    }

    CsrMatrix csr(next_row, next_col, std::move(row_ptr), std::move(col_idx), std::move(values));
    return LpModel(
        std::move(csr),
        std::move(kept_objective),
        std::move(kept_row_lower),
        std::move(kept_row_upper),
        std::move(kept_col_lower),
        std::move(kept_col_upper),
        obj_constant + objective_delta);
  }
};

struct MutableReductionOutcome {
  PresolveStatus status = PresolveStatus::kOk;
  int reductions = 0;
  std::vector<FixedColumnRecord> fixed_columns;
};

MutableReductionOutcome process_empty_rows_mutable(MutablePresolveState& state,
                                                   const PresolveOptions& options) {
  state.seed_empty_rows();

  int reductions = 0;
  while (!state.empty_row_queue.empty()) {
    const int row = state.empty_row_queue.front();
    state.empty_row_queue.pop_front();
    state.empty_row_queued[static_cast<std::size_t>(row)] = 0;
    if (!state.row_is_active_empty(row)) {
      continue;
    }
    if (!empty_row_is_feasible(
            state.row_lower[static_cast<std::size_t>(row)],
            state.row_upper[static_cast<std::size_t>(row)],
            options.feasibility_tolerance)) {
      return MutableReductionOutcome{PresolveStatus::kInfeasible, reductions, {}};
    }
    state.remove_active_row(row);
    ++reductions;
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

MutableReductionOutcome process_singleton_rows_mutable(MutablePresolveState& state,
                                                       const PresolveOptions& options) {
  state.seed_singleton_rows();

  int reductions = 0;
  while (!state.singleton_row_queue.empty()) {
    const int row = state.singleton_row_queue.front();
    state.singleton_row_queue.pop_front();
    state.singleton_row_queued[static_cast<std::size_t>(row)] = 0;
    if (!state.row_is_active_singleton(row)) {
      continue;
    }

    MutableEntry support{-1, 0.0};
    for (const MutableEntry& entry : state.rows.entries(row)) {
      if (state.active_col[static_cast<std::size_t>(entry.index)] != 0 &&
          std::fabs(entry.value) > options.zero_tolerance) {
        support = entry;
        break;
      }
    }
    if (support.index < 0 || std::fabs(support.value) <= options.zero_tolerance) {
      continue;
    }

    const int col = support.index;
    const double a = support.value;
    const double lower = state.row_lower[static_cast<std::size_t>(row)];
    const double upper = state.row_upper[static_cast<std::size_t>(row)];
    const double implied_lower = a > 0.0 ? lower / a : upper / a;
    const double implied_upper = a > 0.0 ? upper / a : lower / a;

    state.set_tighter_col_bounds(col, implied_lower, implied_upper);
    if (state.col_lower[static_cast<std::size_t>(col)] >
        state.col_upper[static_cast<std::size_t>(col)] + options.bound_tolerance) {
      return MutableReductionOutcome{PresolveStatus::kInfeasible, reductions, {}};
    }

    state.remove_active_row(row);
    ++reductions;
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

MutableReductionOutcome process_empty_cols_mutable(MutablePresolveState& state,
                                                   const PresolveOptions& options) {
  state.seed_empty_cols();

  MutableReductionOutcome outcome;
  bool changed = false;
  outcome.fixed_columns.reserve(state.empty_col_queue.remaining_size());
  while (!state.empty_col_queue.empty()) {
    const int col = state.empty_col_queue.front();
    state.empty_col_queue.pop_front();
    state.empty_col_queued[static_cast<std::size_t>(col)] = 0;
    if (!state.col_is_active_empty(col)) {
      continue;
    }

    const double objective = state.objective[static_cast<std::size_t>(col)];
    const double lower = state.col_lower[static_cast<std::size_t>(col)];
    const double upper = state.col_upper[static_cast<std::size_t>(col)];
    if (empty_col_is_unbounded(objective, lower, upper, options.zero_tolerance)) {
      outcome.status = PresolveStatus::kUnbounded;
      return outcome;
    }

    const double value = empty_col_fixed_value(objective, lower, upper, options.zero_tolerance);
    outcome.fixed_columns.push_back(FixedColumnRecord{col, value, objective});
    changed = state.fix_empty_col_without_touch(col, value) || changed;
    ++outcome.reductions;
  }

  if (changed) {
    state.touch_structure_changed();
    state.touch_parallel_signatures();
  }
  return outcome;
}

MutableReductionOutcome process_dual_fix_mutable(MutablePresolveState& state,
                                                 const PresolveOptions& options) {
  state.seed_dual_fix_cols();

  MutableReductionOutcome outcome;
  std::vector<MutableColumnFix> fixes;
  fixes.reserve(state.dual_fix_col_queue.remaining_size());
  outcome.fixed_columns.reserve(state.dual_fix_col_queue.remaining_size());
  while (!state.dual_fix_col_queue.empty()) {
    const int col = state.dual_fix_col_queue.front();
    state.dual_fix_col_queue.pop_front();
    state.dual_fix_col_queued[static_cast<std::size_t>(col)] = 0;
    if (col < 0 || col >= state.num_cols ||
        state.active_col[static_cast<std::size_t>(col)] == 0) {
      continue;
    }

    bool has_down_lock = false;
    bool has_up_lock = false;
    for (const MutableEntry& entry : state.cols.entries(col)) {
      const int row = entry.index;
      if (state.active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      if (entry.value > options.zero_tolerance) {
        has_down_lock =
            has_down_lock || std::isfinite(state.row_lower[static_cast<std::size_t>(row)]);
        has_up_lock =
            has_up_lock || std::isfinite(state.row_upper[static_cast<std::size_t>(row)]);
      } else if (entry.value < -options.zero_tolerance) {
        has_down_lock =
            has_down_lock || std::isfinite(state.row_upper[static_cast<std::size_t>(row)]);
        has_up_lock =
            has_up_lock || std::isfinite(state.row_lower[static_cast<std::size_t>(row)]);
      }
      if (has_down_lock && has_up_lock) {
        break;
      }
    }

    const double objective = state.objective[static_cast<std::size_t>(col)];
    const double lower = state.col_lower[static_cast<std::size_t>(col)];
    const double upper = state.col_upper[static_cast<std::size_t>(col)];
    bool should_fix = false;
    double value = 0.0;
    if (objective > options.zero_tolerance && !has_down_lock) {
      if (!std::isfinite(lower)) {
        outcome.status = PresolveStatus::kUnbounded;
        return outcome;
      }
      should_fix = true;
      value = lower;
    } else if (objective < -options.zero_tolerance && !has_up_lock) {
      if (!std::isfinite(upper)) {
        outcome.status = PresolveStatus::kUnbounded;
        return outcome;
      }
      should_fix = true;
      value = upper;
    } else if (std::fabs(objective) <= options.zero_tolerance) {
      if (!has_down_lock && std::isfinite(lower)) {
        should_fix = true;
        value = lower;
      } else if (!has_up_lock && std::isfinite(upper)) {
        should_fix = true;
        value = upper;
      }
    }
    if (!should_fix) {
      continue;
    }

    outcome.fixed_columns.push_back(FixedColumnRecord{col, value, objective});
    fixes.push_back(MutableColumnFix{col, value});
    ++outcome.reductions;
  }

  state.fix_active_cols_batch(fixes, options.bound_tolerance);
  return outcome;
}

MutableReductionOutcome process_close_bounds_mutable(MutablePresolveState& state,
                                                     const PresolveOptions& options) {
  state.seed_close_bound_cols();

  MutableReductionOutcome outcome;
  std::vector<MutableColumnFix> fixes;
  fixes.reserve(state.close_bound_col_queue.remaining_size());
  outcome.fixed_columns.reserve(state.close_bound_col_queue.remaining_size());
  while (!state.close_bound_col_queue.empty()) {
    const int col = state.close_bound_col_queue.front();
    state.close_bound_col_queue.pop_front();
    state.close_bound_col_queued[static_cast<std::size_t>(col)] = 0;
    if (state.active_col[static_cast<std::size_t>(col)] == 0) {
      continue;
    }
    const double lower = state.col_lower[static_cast<std::size_t>(col)];
    const double upper = state.col_upper[static_cast<std::size_t>(col)];
    if (!is_close_fixed_column(lower, upper, options.bound_tolerance)) {
      continue;
    }

    const double value = 0.5 * (lower + upper);
    const double objective = state.objective[static_cast<std::size_t>(col)];
    outcome.fixed_columns.push_back(FixedColumnRecord{col, value, objective});
    fixes.push_back(MutableColumnFix{col, value});
    ++outcome.reductions;
  }
  state.fix_active_cols_batch(fixes, options.bound_tolerance);
  return outcome;
}

MutableReductionOutcome process_activity_checks_mutable(MutablePresolveState& state,
                                                        const PresolveOptions& options) {
  state.seed_updated_activity_rows();

  int reductions = 0;
  while (!state.updated_activity_row_queue.empty()) {
    const int row = state.updated_activity_row_queue.front();
    state.updated_activity_row_queue.pop_front();
    state.updated_activity_row_queued[static_cast<std::size_t>(row)] = 0;

    if (row < 0 || row >= state.num_rows ||
        state.active_row[static_cast<std::size_t>(row)] == 0 ||
        state.row_size[static_cast<std::size_t>(row)] <= 1) {
      continue;
    }

    const double lower = state.row_lower[static_cast<std::size_t>(row)];
    const double upper = state.row_upper[static_cast<std::size_t>(row)];
    const bool lower_finite = std::isfinite(lower);
    const bool upper_finite = std::isfinite(upper);
    if (lower_finite && upper_finite && std::fabs(upper - lower) <= options.bound_tolerance) {
      continue;
    }

    const CachedRowActivity& activity = state.ensure_row_activity(row);
    double row_min = 0.0;
    double row_max = 0.0;
    if (!activity_sum_value(activity.minimum, row_min) ||
        !activity_sum_value(activity.maximum, row_max)) {
      continue;
    }

    const bool infeasible =
        (lower_finite && row_max < lower - options.bound_tolerance) ||
        (upper_finite && row_min > upper + options.bound_tolerance);
    if (infeasible) {
      return MutableReductionOutcome{PresolveStatus::kInfeasible, reductions, {}};
    }

    const bool lower_implied = !lower_finite || row_min >= lower - options.bound_tolerance;
    const bool upper_implied = !upper_finite || row_max <= upper + options.bound_tolerance;
    if (lower_implied && upper_implied) {
      state.remove_active_row(row);
      ++reductions;
      continue;
    }

    bool side_changed = false;
    double updated_lower = lower;
    double updated_upper = upper;
    if (lower_finite && lower_implied) {
      updated_lower = -std::numeric_limits<double>::infinity();
      side_changed = true;
    } else if (upper_finite && upper_implied) {
      updated_upper = std::numeric_limits<double>::infinity();
      side_changed = true;
    }

    if (side_changed) {
      state.set_row_bounds(row, updated_lower, updated_upper, options.bound_tolerance);
      ++reductions;
    }
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

bool mutable_parallel_row_ratio(const MutablePresolveState& state,
                                int rep_row,
                                int row,
                                double tolerance,
                                double& ratio) {
  const MutableConstEntryRange rep_entries = state.rows.entries(rep_row);
  const MutableConstEntryRange row_entries = state.rows.entries(row);
  if (rep_entries.empty() || rep_entries.size() != row_entries.size()) {
    return false;
  }

  bool ratio_set = false;
  ratio = 0.0;
  for (std::size_t offset = 0; offset < rep_entries.size(); ++offset) {
    if (rep_entries[offset].index != row_entries[offset].index) {
      return false;
    }
    const double rep_value = rep_entries[offset].value;
    const double row_value = row_entries[offset].value;
    if (std::fabs(row_value) <= tolerance) {
      return false;
    }
    if (!ratio_set) {
      ratio = rep_value / row_value;
      ratio_set = true;
    }
    if (std::fabs(rep_value - ratio * row_value) > tolerance) {
      return false;
    }
  }
  return ratio_set && std::fabs(ratio) > tolerance && std::isfinite(ratio);
}

MutableReductionOutcome process_parallel_rows_mutable(MutablePresolveState& state,
                                                      const PresolveOptions& options) {
  int reductions = 0;
  const std::vector<SparsePatternSignature>& signatures = state.row_pattern_signatures();
  for (std::size_t group_begin = 0; group_begin < signatures.size();) {
    std::size_t group_end = group_begin + 1;
    while (group_end < signatures.size() &&
           signatures[group_end].length == signatures[group_begin].length &&
           signatures[group_end].hash == signatures[group_begin].hash &&
           signatures[group_end].value_hash == signatures[group_begin].value_hash) {
      ++group_end;
    }

    if (group_end - group_begin < 2) {
      group_begin = group_end;
      continue;
    }

    std::size_t rep_pos = group_begin;
    while (rep_pos < group_end &&
           state.active_row[static_cast<std::size_t>(signatures[rep_pos].index)] == 0) {
      ++rep_pos;
    }
    if (rep_pos == group_end) {
      group_begin = group_end;
      continue;
    }

    const int rep = signatures[rep_pos].index;
    for (std::size_t row_pos = rep_pos + 1; row_pos < group_end; ++row_pos) {
      const int row = signatures[row_pos].index;
      if (state.active_row[static_cast<std::size_t>(row)] == 0) {
        continue;
      }
      double ratio = 0.0;
      if (!mutable_parallel_row_ratio(state, rep, row, options.zero_tolerance, ratio)) {
        continue;
      }

      double scaled_lower = 0.0;
      double scaled_upper = 0.0;
      scaled_interval(
          state.row_lower[static_cast<std::size_t>(row)],
          state.row_upper[static_cast<std::size_t>(row)],
          ratio,
          scaled_lower,
          scaled_upper);

      const double merged_lower =
          std::max(state.row_lower[static_cast<std::size_t>(rep)], scaled_lower);
      const double merged_upper =
          std::min(state.row_upper[static_cast<std::size_t>(rep)], scaled_upper);
      if (merged_lower > merged_upper + options.feasibility_tolerance) {
        return MutableReductionOutcome{PresolveStatus::kInfeasible, reductions, {}};
      }

      state.set_row_bounds(rep, merged_lower, merged_upper, options.bound_tolerance);
      state.remove_active_row(row);
      ++reductions;
    }
    group_begin = group_end;
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

bool mutable_parallel_col_ratio(const MutablePresolveState& state,
                               int rep_col,
                               int col,
                               double tolerance,
                               double& ratio) {
  const MutableConstEntryRange rep_entries = state.cols.entries(rep_col);
  const MutableConstEntryRange col_entries = state.cols.entries(col);
  if (rep_entries.empty() || rep_entries.size() != col_entries.size()) {
    return false;
  }

  bool ratio_set = false;
  ratio = 0.0;
  for (std::size_t offset = 0; offset < rep_entries.size(); ++offset) {
    if (rep_entries[offset].index != col_entries[offset].index) {
      return false;
    }
    const double rep_value = rep_entries[offset].value;
    const double col_value = col_entries[offset].value;
    if (std::fabs(col_value) <= tolerance) {
      return false;
    }
    if (!ratio_set) {
      ratio = col_value / rep_value;
      ratio_set = true;
    }
    if (std::fabs(col_value - ratio * rep_value) > tolerance) {
      return false;
    }
  }
  return ratio_set && std::fabs(ratio) > tolerance && std::isfinite(ratio);
}

double merged_parallel_col_lower(double target_lower,
                                double source_lower,
                                double source_upper,
                                double ratio);

double merged_parallel_col_upper(double target_upper,
                                double source_lower,
                                double source_upper,
                                double ratio);

MutableReductionOutcome process_parallel_cols_mutable(MutablePresolveState& state,
                                                     const PresolveOptions& options) {
  int reductions = 0;
  MutableReductionOutcome outcome;
  const std::vector<SparsePatternSignature>& signatures = state.col_pattern_signatures();

  for (std::size_t group_begin = 0; group_begin < signatures.size();) {
    std::size_t group_end = group_begin + 1;
    while (group_end < signatures.size() &&
           signatures[group_end].length == signatures[group_begin].length &&
           signatures[group_end].hash == signatures[group_begin].hash &&
           signatures[group_end].value_hash == signatures[group_begin].value_hash) {
      ++group_end;
    }

    if (group_end - group_begin < 2) {
      group_begin = group_end;
      continue;
    }

    for (std::size_t target_pos = group_begin; target_pos + 1 < group_end; ++target_pos) {
      const int target = signatures[target_pos].index;
      if (state.active_col[static_cast<std::size_t>(target)] == 0) {
        continue;
      }

      for (std::size_t source_pos = target_pos + 1; source_pos < group_end; ++source_pos) {
        const int source = signatures[source_pos].index;
        if (state.active_col[static_cast<std::size_t>(source)] == 0) {
          continue;
        }
        double ratio = 0.0;
        if (!mutable_parallel_col_ratio(state, target, source, options.zero_tolerance, ratio)) {
          continue;
        }

        const double obj_gap =
            state.objective[static_cast<std::size_t>(source)] -
            ratio * state.objective[static_cast<std::size_t>(target)];
        if (std::fabs(obj_gap) <= options.zero_tolerance) {
          state.set_col_bounds(
              target,
              merged_parallel_col_lower(
                  state.col_lower[static_cast<std::size_t>(target)],
                  state.col_lower[static_cast<std::size_t>(source)],
                  state.col_upper[static_cast<std::size_t>(source)],
                  ratio),
              merged_parallel_col_upper(
                  state.col_upper[static_cast<std::size_t>(target)],
                  state.col_lower[static_cast<std::size_t>(source)],
                  state.col_upper[static_cast<std::size_t>(source)],
                  ratio));
          state.remove_active_col(source);
          ++reductions;
          continue;
        }

        bool fix_source_to_lower = false;
        bool fix_source_to_upper = false;
        bool fix_target_to_lower = false;
        bool fix_target_to_upper = false;
        const double target_lower = state.col_lower[static_cast<std::size_t>(target)];
        const double target_upper = state.col_upper[static_cast<std::size_t>(target)];
        const double source_lower = state.col_lower[static_cast<std::size_t>(source)];
        const double source_upper = state.col_upper[static_cast<std::size_t>(source)];
        if (obj_gap > options.zero_tolerance) {
          if (ratio > 0.0) {
            fix_source_to_lower = !std::isfinite(target_upper);
            fix_target_to_upper = !std::isfinite(source_lower);
          } else {
            fix_source_to_lower = !std::isfinite(target_lower);
            fix_target_to_lower = !std::isfinite(source_lower);
          }
        } else {
          if (ratio > 0.0) {
            fix_source_to_upper = !std::isfinite(target_lower);
            fix_target_to_lower = !std::isfinite(source_upper);
          } else {
            fix_source_to_upper = !std::isfinite(target_upper);
            fix_target_to_upper = !std::isfinite(source_upper);
          }
        }

        auto fix_column = [&](int col, double value) {
          outcome.fixed_columns.push_back(
              FixedColumnRecord{col, value, state.objective[static_cast<std::size_t>(col)]});
          state.fix_active_col(col, value);
          ++reductions;
        };

        if (fix_source_to_lower) {
          if (!std::isfinite(source_lower)) {
            outcome.status = PresolveStatus::kUnbounded;
            return outcome;
          }
          fix_column(source, source_lower);
          continue;
        }
        if (fix_source_to_upper) {
          if (!std::isfinite(source_upper)) {
            outcome.status = PresolveStatus::kUnbounded;
            return outcome;
          }
          fix_column(source, source_upper);
          continue;
        }
        if (fix_target_to_lower) {
          if (!std::isfinite(target_lower)) {
            outcome.status = PresolveStatus::kUnbounded;
            return outcome;
          }
          fix_column(target, target_lower);
          break;
        }
        if (fix_target_to_upper) {
          if (!std::isfinite(target_upper)) {
            outcome.status = PresolveStatus::kUnbounded;
            return outcome;
          }
          fix_column(target, target_upper);
          break;
        }
      }
    }
    group_begin = group_end;
  }

  outcome.reductions = reductions;
  return outcome;
}

MutableReductionOutcome process_singleton_cols_eq(MutablePresolveState& state,
                                                  const PresolveOptions& options);

MutableReductionOutcome process_doubleton_eq(MutablePresolveState& state,
                                             const PresolveOptions& options);

struct DoubletonCandidate {
  int row = -1;
  int elim_col = -1;
  int keep_col = -1;
};

int mutable_doubleton_fill_in_proxy(const MutablePresolveState& state,
                                    int keep_col,
                                    int elim_col) {
  MutableConstEntryRange keep_entries = state.cols.entries(keep_col);
  MutableConstEntryRange elim_entries = state.cols.entries(elim_col);
  if (keep_entries.empty() || elim_entries.empty()) {
    return -1;
  }

  int fill_in = -1;
  std::size_t keep_pos = 0;
  std::size_t elim_pos = 0;
  while (keep_pos < keep_entries.size() && elim_pos < elim_entries.size()) {
    const int keep_row = keep_entries[keep_pos].index;
    const int elim_row = elim_entries[elim_pos].index;
    if (keep_row == elim_row) {
      ++keep_pos;
      ++elim_pos;
    } else if (elim_row < keep_row) {
      ++elim_pos;
      ++fill_in;
    } else {
      ++keep_pos;
    }
  }
  fill_in += static_cast<int>(elim_entries.size() - elim_pos);
  return fill_in;
}

bool build_doubleton_candidate(const MutablePresolveState& state,
                               const PresolveOptions& options,
                               int row,
                               DoubletonCandidate& candidate) {
  if (!state.row_is_active_doubleton_eq(row, options.bound_tolerance)) {
    return false;
  }

  const MutableConstEntryRange row_entries = state.rows.entries(row);
  const int col1 = row_entries[0].index;
  const int col2 = row_entries[1].index;
  const double val1 = row_entries[0].value;
  const double val2 = row_entries[1].value;
  if (state.active_col[static_cast<std::size_t>(col1)] == 0 ||
      state.active_col[static_cast<std::size_t>(col2)] == 0) {
    return false;
  }

  int elim_col = -1;
  int keep_col = -1;
  double elim_val = 0.0;
  double keep_val = 0.0;
  choose_doubleton_columns(
      col1,
      val1,
      col2,
      val2,
      state.col_size,
      options.bound_tolerance,
      elim_col,
      elim_val,
      keep_col,
      keep_val);
  if (!acceptable_doubleton_pivot(keep_val, elim_val)) {
    return false;
  }
  if (mutable_doubleton_fill_in_proxy(state, keep_col, elim_col) >
      options.doubleton_eq_max_fill_in_proxy) {
    return false;
  }

  candidate = DoubletonCandidate{row, elim_col, keep_col};
  return true;
}

MutableReductionOutcome apply_doubleton_candidate(MutablePresolveState& state,
                                                  const PresolveOptions& options,
                                                  int row) {
  if (!state.row_is_active_doubleton_eq(row, options.bound_tolerance)) {
    return MutableReductionOutcome{};
  }

  const MutableConstEntryRange row_entries = state.rows.entries(row);
  const int col1 = row_entries[0].index;
  const int col2 = row_entries[1].index;
  const double val1 = row_entries[0].value;
  const double val2 = row_entries[1].value;
  if (state.active_col[static_cast<std::size_t>(col1)] == 0 ||
      state.active_col[static_cast<std::size_t>(col2)] == 0) {
    return MutableReductionOutcome{};
  }

  int elim_col = -1;
  int keep_col = -1;
  double elim_val = 0.0;
  double keep_val = 0.0;
  choose_doubleton_columns(
      col1,
      val1,
      col2,
      val2,
      state.col_size,
      options.bound_tolerance,
      elim_col,
      elim_val,
      keep_col,
      keep_val);
  if (!acceptable_doubleton_pivot(keep_val, elim_val)) {
    return MutableReductionOutcome{};
  }

  const double rhs = state.row_upper[static_cast<std::size_t>(row)];
  const double alpha = -keep_val / elim_val;
  const double beta = rhs / elim_val;
  double mapped_lower = 0.0;
  double mapped_upper = 0.0;
  doubleton_mapped_interval(
      rhs,
      keep_val,
      elim_val,
      state.col_lower[static_cast<std::size_t>(elim_col)],
      state.col_upper[static_cast<std::size_t>(elim_col)],
      mapped_lower,
      mapped_upper);

  const double old_keep_lower = state.col_lower[static_cast<std::size_t>(keep_col)];
  const double old_keep_upper = state.col_upper[static_cast<std::size_t>(keep_col)];
  const double updated_lower = std::max(old_keep_lower, mapped_lower);
  const double updated_upper = std::min(old_keep_upper, mapped_upper);
  if (updated_lower > updated_upper + options.bound_tolerance) {
    return MutableReductionOutcome{PresolveStatus::kInfeasible, 0, {}};
  }

  state.set_tighter_col_bounds(keep_col, updated_lower, updated_upper);

  const double elim_objective = state.objective[static_cast<std::size_t>(elim_col)];
  state.objective[static_cast<std::size_t>(keep_col)] += elim_objective * alpha;
  state.objective_delta += elim_objective * beta;

  state.remove_doubleton_pivot_row(row, elim_col, keep_col);
  while (!state.cols.empty(elim_col)) {
    const MutableEntry elim_entry = state.cols.back(elim_col);
    state.cols.pop_back(elim_col);
    --state.col_size[static_cast<std::size_t>(elim_col)];
    const int affected_row = elim_entry.index;
    if (state.active_row[static_cast<std::size_t>(affected_row)] == 0) {
      continue;
    }
    double elim_coeff = 0.0;
    if (!state.substitute_doubleton_row_entry(
            affected_row,
            elim_col,
            keep_col,
            alpha * elim_entry.value,
            &elim_coeff)) {
      continue;
    }
    state.shift_row_bounds_preserving_locks(
        affected_row,
        elim_coeff * beta,
        options.bound_tolerance);
  }

  state.touch_structure_changed();
  state.touch_parallel_signatures();
  state.deactivate_empty_col(elim_col);
  state.enqueue_singleton_dual_col(col1);
  state.enqueue_singleton_dual_col(col2);
  state.enqueue_singleton_dual_col(keep_col);
  state.enqueue_singleton_col(col1);
  state.enqueue_singleton_col(col2);
  state.enqueue_singleton_col(keep_col);
  ++state.doubleton_eq_reductions_total;
  return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
}

MutableReductionOutcome process_doubleton_eq_batch(MutablePresolveState& state,
                                                   const PresolveOptions& options) {
  const bool profile_batch = std::getenv("GPU_PRESOLVER_DOUBLETON_BATCH_PROFILE") != nullptr;
  int reductions = 0;
  while (true) {
    if (options.doubleton_eq_max_reductions >= 0 &&
        state.doubleton_eq_reductions_total >= options.doubleton_eq_max_reductions) {
      break;
    }

    std::vector<DoubletonCandidate> acceptable;
    acceptable.reserve(static_cast<std::size_t>(state.num_rows));
    for (int row = 0; row < state.num_rows; ++row) {
      DoubletonCandidate candidate;
      if (build_doubleton_candidate(state, options, row, candidate)) {
        acceptable.push_back(candidate);
      }
    }
    if (acceptable.empty()) {
      break;
    }

    std::vector<unsigned char> active_mask(acceptable.size(), 1);
    std::vector<unsigned char> selected_mask(acceptable.size(), 0);
    std::vector<unsigned char> blocked_col(static_cast<std::size_t>(state.num_cols), 0);
    std::vector<int> col_owner(static_cast<std::size_t>(state.num_cols), std::numeric_limits<int>::max());
    constexpr int kBatchMatchingRounds = 8;
    for (int round = 0; round < kBatchMatchingRounds; ++round) {
      std::fill(col_owner.begin(), col_owner.end(), std::numeric_limits<int>::max());
      for (std::size_t i = 0; i < acceptable.size(); ++i) {
        if (active_mask[i] == 0) {
          continue;
        }
        const DoubletonCandidate& candidate = acceptable[i];
        if (blocked_col[static_cast<std::size_t>(candidate.keep_col)] != 0 ||
            blocked_col[static_cast<std::size_t>(candidate.elim_col)] != 0) {
          continue;
        }
        col_owner[static_cast<std::size_t>(candidate.keep_col)] =
            std::min(col_owner[static_cast<std::size_t>(candidate.keep_col)], candidate.row);
        col_owner[static_cast<std::size_t>(candidate.elim_col)] =
            std::min(col_owner[static_cast<std::size_t>(candidate.elim_col)], candidate.row);
      }

      for (std::size_t i = 0; i < acceptable.size(); ++i) {
        if (active_mask[i] == 0) {
          continue;
        }
        const DoubletonCandidate& candidate = acceptable[i];
        if (blocked_col[static_cast<std::size_t>(candidate.keep_col)] == 0 &&
            blocked_col[static_cast<std::size_t>(candidate.elim_col)] == 0 &&
            col_owner[static_cast<std::size_t>(candidate.keep_col)] == candidate.row &&
            col_owner[static_cast<std::size_t>(candidate.elim_col)] == candidate.row) {
          selected_mask[i] = 1;
          active_mask[i] = 0;
          blocked_col[static_cast<std::size_t>(candidate.keep_col)] = 1;
          blocked_col[static_cast<std::size_t>(candidate.elim_col)] = 1;
        }
      }
    }

    std::vector<DoubletonCandidate> selected;
    selected.reserve(acceptable.size());
    for (std::size_t i = 0; i < acceptable.size(); ++i) {
      if (selected_mask[i] != 0) {
        selected.push_back(acceptable[i]);
      }
    }
    if (profile_batch) {
      std::cout << ">>> [doubleton_eq batch] acceptable=" << acceptable.size()
                << " selected=" << selected.size() << '\n';
    }
    if (selected.empty()) {
      break;
    }

    const double selected_ratio =
        static_cast<double>(selected.size()) / static_cast<double>(acceptable.size());
    if (static_cast<int>(selected.size()) < options.doubleton_eq_min_selected_per_batch ||
        selected_ratio < options.doubleton_eq_min_selected_ratio) {
      break;
    }

    int applied_this_round = 0;
    for (const DoubletonCandidate& candidate : selected) {
      if (options.doubleton_eq_max_reductions >= 0 &&
          state.doubleton_eq_reductions_total >= options.doubleton_eq_max_reductions) {
        break;
      }
      MutableReductionOutcome outcome = apply_doubleton_candidate(state, options, candidate.row);
      if (outcome.status != PresolveStatus::kOk) {
        return MutableReductionOutcome{outcome.status, reductions, {}};
      }
      reductions += outcome.reductions;
      applied_this_round += outcome.reductions;
    }
    if (applied_this_round == 0) {
      break;
    }
  }
  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

MutableReductionOutcome process_singleton_col_eq_candidate(MutablePresolveState& state,
                                                           const PresolveOptions& options,
                                                           int col,
                                                           int row,
                                                           double coefficient) {
  if (state.active_row[static_cast<std::size_t>(row)] == 0 ||
      state.row_size[static_cast<std::size_t>(row)] <= 1 ||
      !is_equality_row(state.row_lower[static_cast<std::size_t>(row)],
                       state.row_upper[static_cast<std::size_t>(row)],
                       options.bound_tolerance)) {
    return MutableReductionOutcome{};
  }

  const CachedRowActivity& activity = state.ensure_row_activity(row);
  const double current_min = term_min(
      coefficient,
      state.col_lower[static_cast<std::size_t>(col)],
      state.col_upper[static_cast<std::size_t>(col)]);
  const double current_max = term_max(
      coefficient,
      state.col_lower[static_cast<std::size_t>(col)],
      state.col_upper[static_cast<std::size_t>(col)]);
  double rest_min = 0.0;
  double rest_max = 0.0;
  if (!activity_value_excluding(activity.minimum, current_min, rest_min) ||
      !activity_value_excluding(activity.maximum, current_max, rest_max)) {
    return MutableReductionOutcome{};
  }

  const double rhs = state.row_upper[static_cast<std::size_t>(row)];
  const double x1 = (rhs - rest_min) / coefficient;
  const double x2 = (rhs - rest_max) / coefficient;
  const double implied_lower = std::min(x1, x2);
  const double implied_upper = std::max(x1, x2);
  const bool free_above = singleton_col_eq_free_from_above(
      implied_upper,
      state.col_upper[static_cast<std::size_t>(col)],
      options.bound_tolerance);
  const bool free_below = singleton_col_eq_free_from_below(
      implied_lower,
      state.col_lower[static_cast<std::size_t>(col)],
      options.bound_tolerance);
  if (!free_above && !free_below) {
    return MutableReductionOutcome{};
  }
  if (free_above && !free_below &&
      !std::isfinite(state.col_lower[static_cast<std::size_t>(col)])) {
    return MutableReductionOutcome{};
  }
  if (!free_above && free_below &&
      !std::isfinite(state.col_upper[static_cast<std::size_t>(col)])) {
    return MutableReductionOutcome{};
  }

  const double objective = state.objective[static_cast<std::size_t>(col)];
  state.objective_delta += objective * rhs / coefficient;
  std::vector<int> row_cols;
  row_cols.reserve(state.rows.entries(row).size());
  for (const MutableEntry& entry : state.rows.entries(row)) {
    row_cols.push_back(entry.index);
    if (entry.index == col) {
      continue;
    }
    state.objective[static_cast<std::size_t>(entry.index)] -=
        objective * entry.value / coefficient;
  }

  if (free_above && free_below) {
    state.remove_active_row(row);
    state.deactivate_empty_col(col);
    for (int affected_col : row_cols) {
      if (affected_col != col) {
        state.enqueue_singleton_dual_col(affected_col);
        state.enqueue_singleton_col(affected_col);
      }
    }
    return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
  }

  double updated_lower = state.row_lower[static_cast<std::size_t>(row)];
  double updated_upper = state.row_upper[static_cast<std::size_t>(row)];
  if (free_above) {
    const double shifted_rhs =
        rhs - coefficient * state.col_lower[static_cast<std::size_t>(col)];
    if (coefficient < 0.0) {
      updated_lower = shifted_rhs;
      updated_upper = std::numeric_limits<double>::infinity();
    } else {
      updated_lower = -std::numeric_limits<double>::infinity();
      updated_upper = shifted_rhs;
    }
  } else {
    const double shifted_rhs =
        rhs - coefficient * state.col_upper[static_cast<std::size_t>(col)];
    if (coefficient > 0.0) {
      updated_lower = shifted_rhs;
      updated_upper = std::numeric_limits<double>::infinity();
    } else {
      updated_lower = -std::numeric_limits<double>::infinity();
      updated_upper = shifted_rhs;
    }
  }

  state.set_row_bounds(row, updated_lower, updated_upper, options.bound_tolerance);
  state.remove_row_col_entry(row, col, nullptr);
  state.deactivate_empty_col(col);
  return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
}

MutableReductionOutcome substitute_singleton_col_from_active_side(MutablePresolveState& state,
                                                                  int row,
                                                                  int col,
                                                                  double coefficient,
                                                                  double active_side) {
  const double objective = state.objective[static_cast<std::size_t>(col)];
  state.objective_delta += objective * active_side / coefficient;

  std::vector<int> row_cols;
  row_cols.reserve(state.rows.entries(row).size());
  for (const MutableEntry& entry : state.rows.entries(row)) {
    row_cols.push_back(entry.index);
    if (entry.index == col) {
      continue;
    }
    state.objective[static_cast<std::size_t>(entry.index)] -=
        objective * entry.value / coefficient;
  }

  state.remove_active_row(row);
  state.deactivate_empty_col(col);
  for (int affected_col : row_cols) {
    if (affected_col != col) {
      state.enqueue_singleton_dual_col(affected_col);
      state.enqueue_singleton_col(affected_col);
    }
  }
  return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
}

MutableReductionOutcome process_singleton_col_ineq_candidate(MutablePresolveState& state,
                                                             const PresolveOptions& options,
                                                             int col,
                                                             int row,
                                                             double coefficient) {
  if (state.active_row[static_cast<std::size_t>(row)] == 0 ||
      state.row_size[static_cast<std::size_t>(row)] <= 1 ||
      is_equality_row(
          state.row_lower[static_cast<std::size_t>(row)],
          state.row_upper[static_cast<std::size_t>(row)],
          options.bound_tolerance)) {
    return MutableReductionOutcome{};
  }

  const double objective = state.objective[static_cast<std::size_t>(col)];
  const double row_lower = state.row_lower[static_cast<std::size_t>(row)];
  const double row_upper = state.row_upper[static_cast<std::size_t>(row)];
  if (singleton_col_direct_unbounded(
          objective,
          coefficient,
          row_lower,
          row_upper,
          state.col_lower[static_cast<std::size_t>(col)],
          state.col_upper[static_cast<std::size_t>(col)],
          options.zero_tolerance)) {
    return MutableReductionOutcome{PresolveStatus::kUnbounded, 0, {}};
  }

  const CachedRowActivity& activity = state.ensure_row_activity(row);
  const double current_min = term_min(
      coefficient,
      state.col_lower[static_cast<std::size_t>(col)],
      state.col_upper[static_cast<std::size_t>(col)]);
  const double current_max = term_max(
      coefficient,
      state.col_lower[static_cast<std::size_t>(col)],
      state.col_upper[static_cast<std::size_t>(col)]);
  double rest_min = 0.0;
  double rest_max = 0.0;
  if (!activity_value_excluding(activity.minimum, current_min, rest_min) ||
      !activity_value_excluding(activity.maximum, current_max, rest_max)) {
    return MutableReductionOutcome{};
  }

  const bool free_above = singleton_col_implied_free_from_above(
      coefficient,
      row_lower,
      row_upper,
      state.col_upper[static_cast<std::size_t>(col)],
      rest_min,
      rest_max,
      options.bound_tolerance);
  const bool free_below = singleton_col_implied_free_from_below(
      coefficient,
      row_lower,
      row_upper,
      state.col_lower[static_cast<std::size_t>(col)],
      rest_min,
      rest_max,
      options.bound_tolerance);

  if (free_above && free_below) {
    const double active_side = singleton_col_active_side(
        objective,
        coefficient,
        row_lower,
        row_upper,
        options.zero_tolerance);
    if (!std::isfinite(active_side)) {
      return MutableReductionOutcome{};
    }
    if (!options.enable_singleton_cols_eq) {
      state.set_row_bounds(row, active_side, active_side, options.bound_tolerance);
      state.enqueue_singleton_col(col);
      return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
    }
    return substitute_singleton_col_from_active_side(
        state,
        row,
        col,
        coefficient,
        active_side);
  }

  double active_side = std::numeric_limits<double>::quiet_NaN();
  if ((objective < -options.zero_tolerance && coefficient > 0.0 && free_above) ||
      (objective > options.zero_tolerance && coefficient < 0.0 && free_below)) {
    active_side = row_upper;
  } else if ((objective > options.zero_tolerance && coefficient > 0.0 && free_below) ||
             (objective < -options.zero_tolerance && coefficient < 0.0 && free_above)) {
    active_side = row_lower;
  } else {
    return MutableReductionOutcome{};
  }

  if (!std::isfinite(active_side)) {
    return MutableReductionOutcome{};
  }

  state.set_row_bounds(row, active_side, active_side, options.bound_tolerance);
  state.enqueue_singleton_col(col);
  return MutableReductionOutcome{PresolveStatus::kOk, 1, {}};
}

MutableReductionOutcome process_singleton_cols_combined_mutable(MutablePresolveState& state,
                                                                const PresolveOptions& options) {
  MutableReductionOutcome combined;
  state.seed_singleton_cols();

  auto append = [&](MutableReductionOutcome&& outcome) {
    if (outcome.status != PresolveStatus::kOk) {
      combined.status = outcome.status;
      return false;
    }
    combined.reductions += outcome.reductions;
    combined.fixed_columns.reserve(
        combined.fixed_columns.size() + outcome.fixed_columns.size());
    combined.fixed_columns.insert(
        combined.fixed_columns.end(),
        std::make_move_iterator(outcome.fixed_columns.begin()),
        std::make_move_iterator(outcome.fixed_columns.end()));
    return true;
  };

  for (;;) {
    const int reductions_before = combined.reductions;

    while (!state.singleton_col_queue.empty()) {
      const int col = state.singleton_col_queue.front();
      state.singleton_col_queue.pop_front();
      state.singleton_col_queued[static_cast<std::size_t>(col)] = 0;

      if (!state.col_is_active_singleton(col)) {
        continue;
      }

      MutableEntry support{-1, 0.0};
      for (const MutableEntry& entry : state.cols.entries(col)) {
        if (state.active_row[static_cast<std::size_t>(entry.index)] != 0 &&
            std::fabs(entry.value) > options.zero_tolerance) {
          support = entry;
          break;
        }
      }
      if (support.index < 0) {
        continue;
      }

      const int row = support.index;
      if (state.row_size[static_cast<std::size_t>(row)] <= 1) {
        continue;
      }

      const bool equality =
          is_equality_row(
              state.row_lower[static_cast<std::size_t>(row)],
              state.row_upper[static_cast<std::size_t>(row)],
              options.bound_tolerance);
      if (equality) {
        if (options.enable_singleton_cols_eq &&
            !append(process_singleton_col_eq_candidate(
                state,
                options,
                col,
                row,
                support.value))) {
          return combined;
        }
      } else if (options.enable_singleton_cols_dual_infer &&
                 !append(process_singleton_col_ineq_candidate(
                     state,
                     options,
                     col,
                     row,
                     support.value))) {
        return combined;
      }
    }

    if (combined.reductions == reductions_before) {
      break;
    }
  }
  return combined;
}

MutableReductionOutcome process_primal_propagation_dirty(MutablePresolveState& state,
                                                         const PresolveOptions& options) {
  struct PropagationTerm {
    int col;
    double value;
  };
  while (!state.next_dirty_row_queue.empty()) {
    const int row = state.next_dirty_row_queue.front();
    state.next_dirty_row_queue.pop_front();
    if (state.active_row[static_cast<std::size_t>(row)] == 0 ||
        state.dirty_row_state[static_cast<std::size_t>(row)] != 3) {
      continue;
    }
    state.dirty_row_state[static_cast<std::size_t>(row)] = 1;
    state.dirty_row_queue.push_back(row);
  }
  state.seed_propagation_rows();
  int tightened = 0;
  std::vector<int> propagated_rows;
  std::vector<PropagationTerm> terms;
  state.propagation_active = true;
  state.activity_cache_active = true;
  auto finish = [&](PresolveStatus status) {
    state.propagation_active = false;
    for (int propagated_row : propagated_rows) {
      if (state.dirty_row_state[static_cast<std::size_t>(propagated_row)] == 2) {
        state.dirty_row_state[static_cast<std::size_t>(propagated_row)] = 0;
      }
    }
    return MutableReductionOutcome{status, tightened, {}};
  };

  while (!state.dirty_row_queue.empty()) {
    const int row = state.dirty_row_queue.front();
    state.dirty_row_queue.pop_front();
    if (state.dirty_row_state[static_cast<std::size_t>(row)] != 1) {
      continue;
    }
    state.dirty_row_state[static_cast<std::size_t>(row)] = 2;
    propagated_rows.push_back(row);

    if (state.active_row[static_cast<std::size_t>(row)] == 0 ||
        state.row_size[static_cast<std::size_t>(row)] <= 0) {
      continue;
    }

    const double row_lower = state.row_lower[static_cast<std::size_t>(row)];
    const double row_upper = state.row_upper[static_cast<std::size_t>(row)];
    if (!std::isfinite(row_lower) && !std::isfinite(row_upper)) {
      continue;
    }

    CachedRowActivity activity;
    terms.clear();
    for (const MutableEntry& entry : state.rows.entries(row)) {
      const int col = entry.index;
      if (state.active_col[static_cast<std::size_t>(col)] == 0 ||
          std::fabs(entry.value) <= options.zero_tolerance) {
        continue;
      }
      add_activity_value(
          activity.minimum,
          term_min(
              entry.value,
              state.col_lower[static_cast<std::size_t>(col)],
              state.col_upper[static_cast<std::size_t>(col)]));
      add_activity_value(
          activity.maximum,
          term_max(
              entry.value,
              state.col_lower[static_cast<std::size_t>(col)],
              state.col_upper[static_cast<std::size_t>(col)]));
      terms.push_back(PropagationTerm{col, entry.value});
    }
    state.row_activity[static_cast<std::size_t>(row)] = activity;
    state.activity_row_dirty[static_cast<std::size_t>(row)] = 0;

    const ActivitySum row_min_sum = activity.minimum;
    const ActivitySum row_max_sum = activity.maximum;
    const int min_inf_count = row_min_sum.negative_infinity + row_min_sum.positive_infinity;
    const int max_inf_count = row_max_sum.negative_infinity + row_max_sum.positive_infinity;
    const bool use_all_min_terms =
        std::isfinite(row_upper) && !row_min_sum.has_nan && min_inf_count == 0;
    const bool use_all_max_terms =
        std::isfinite(row_lower) && !row_max_sum.has_nan && max_inf_count == 0;
    const bool use_single_min_term =
        std::isfinite(row_upper) && !row_min_sum.has_nan && min_inf_count == 1;
    const bool use_single_max_term =
        std::isfinite(row_lower) && !row_max_sum.has_nan && max_inf_count == 1;
    if (!use_all_min_terms && !use_all_max_terms &&
        !use_single_min_term && !use_single_max_term) {
      continue;
    }

    for (const PropagationTerm& term : terms) {
      const int col = term.col;
      const double a = term.value;
      const double current_min = term_min(
          a,
          state.col_lower[static_cast<std::size_t>(col)],
          state.col_upper[static_cast<std::size_t>(col)]);
      const double current_max = term_max(
          a,
          state.col_lower[static_cast<std::size_t>(col)],
          state.col_upper[static_cast<std::size_t>(col)]);
      double rest_min = 0.0;
      double rest_max = 0.0;
      const bool rest_min_finite =
          (use_all_min_terms || (use_single_min_term && std::isinf(current_min))) &&
          finite_activity_excluding(row_min_sum, current_min, rest_min);
      const bool rest_max_finite =
          (use_all_max_terms || (use_single_max_term && std::isinf(current_max))) &&
          finite_activity_excluding(row_max_sum, current_max, rest_max);
      if (!rest_min_finite && !rest_max_finite) {
        continue;
      }

      double candidate_lower = state.col_lower[static_cast<std::size_t>(col)];
      double candidate_upper = state.col_upper[static_cast<std::size_t>(col)];
      bool col_changed = false;
      auto accept_lower = [&](double implied_lower) {
        return implied_lower > candidate_lower;
      };
      auto accept_upper = [&](double implied_upper) {
        return implied_upper < candidate_upper;
      };
      if (a > 0.0) {
        if (std::isfinite(row_lower) && rest_max_finite) {
          const double implied_lower = (row_lower - rest_max) / a;
          if (accept_lower(implied_lower)) {
            candidate_lower = implied_lower;
            col_changed = true;
          }
        }
        if (std::isfinite(row_upper) && rest_min_finite) {
          const double implied_upper = (row_upper - rest_min) / a;
          if (accept_upper(implied_upper)) {
            candidate_upper = implied_upper;
            col_changed = true;
          }
        }
      } else {
        if (std::isfinite(row_upper) && rest_min_finite) {
          const double implied_lower = (row_upper - rest_min) / a;
          if (accept_lower(implied_lower)) {
            candidate_lower = implied_lower;
            col_changed = true;
          }
        }
        if (std::isfinite(row_lower) && rest_max_finite) {
          const double implied_upper = (row_lower - rest_max) / a;
          if (accept_upper(implied_upper)) {
            candidate_upper = implied_upper;
            col_changed = true;
          }
        }
      }

      if (col_changed) {
        state.set_tighter_col_bounds(col, candidate_lower, candidate_upper);
      }
      if (state.col_lower[static_cast<std::size_t>(col)] >
          state.col_upper[static_cast<std::size_t>(col)] + options.feasibility_tolerance) {
        return finish(PresolveStatus::kInfeasible);
      }
      if (col_changed) {
        ++tightened;
      }
    }
  }

  return finish(PresolveStatus::kOk);
}

  MutableReductionOutcome process_singleton_cols_eq(MutablePresolveState& state,
                                                  const PresolveOptions& options) {
  state.seed_singleton_cols();

  int reductions = 0;
  while (!state.singleton_col_queue.empty()) {
    const int col = state.singleton_col_queue.front();
    state.singleton_col_queue.pop_front();
    state.singleton_col_queued[static_cast<std::size_t>(col)] = 0;

    if (!state.col_is_active_singleton(col)) {
      continue;
    }

    const MutableEntry support = state.cols.front(col);
    const int row = support.index;
    const double coefficient = support.value;
    if (state.active_row[static_cast<std::size_t>(row)] == 0 ||
        state.row_size[static_cast<std::size_t>(row)] <= 1 ||
        !is_equality_row(state.row_lower[static_cast<std::size_t>(row)],
                         state.row_upper[static_cast<std::size_t>(row)],
                         options.bound_tolerance)) {
      continue;
    }

    const CachedRowActivity& activity = state.ensure_row_activity(row);
    const double current_min = term_min(
        coefficient,
        state.col_lower[static_cast<std::size_t>(col)],
        state.col_upper[static_cast<std::size_t>(col)]);
    const double current_max = term_max(
        coefficient,
        state.col_lower[static_cast<std::size_t>(col)],
        state.col_upper[static_cast<std::size_t>(col)]);
    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!activity_value_excluding(activity.minimum, current_min, rest_min) ||
        !activity_value_excluding(activity.maximum, current_max, rest_max)) {
      continue;
    }

    const double rhs = state.row_upper[static_cast<std::size_t>(row)];
    const double x1 = (rhs - rest_min) / coefficient;
    const double x2 = (rhs - rest_max) / coefficient;
    const double implied_lower = std::min(x1, x2);
    const double implied_upper = std::max(x1, x2);
    const bool free_above = singleton_col_eq_free_from_above(
        implied_upper,
        state.col_upper[static_cast<std::size_t>(col)],
        options.bound_tolerance);
    const bool free_below = singleton_col_eq_free_from_below(
        implied_lower,
        state.col_lower[static_cast<std::size_t>(col)],
        options.bound_tolerance);
    if (!free_above && !free_below) {
      continue;
    }

    const double objective = state.objective[static_cast<std::size_t>(col)];
    state.objective_delta += objective * rhs / coefficient;
    std::vector<int> row_cols;
    row_cols.reserve(state.rows.entries(row).size());
    for (const MutableEntry& entry : state.rows.entries(row)) {
      row_cols.push_back(entry.index);
      if (entry.index == col) {
        continue;
      }
      state.objective[static_cast<std::size_t>(entry.index)] -=
          objective * entry.value / coefficient;
    }

    if (free_above && free_below) {
      state.remove_active_row(row);
      state.deactivate_empty_col(col);
      for (int affected_col : row_cols) {
        if (affected_col != col) {
          state.enqueue_singleton_dual_col(affected_col);
          state.enqueue_singleton_col(affected_col);
        }
      }
    } else {
      double updated_lower = state.row_lower[static_cast<std::size_t>(row)];
      double updated_upper = state.row_upper[static_cast<std::size_t>(row)];
      if (free_above) {
        if (!std::isfinite(state.col_lower[static_cast<std::size_t>(col)])) {
          continue;
        }
        const double shifted_rhs =
            rhs - coefficient * state.col_lower[static_cast<std::size_t>(col)];
        if (coefficient < 0.0) {
          updated_lower = shifted_rhs;
          updated_upper = std::numeric_limits<double>::infinity();
        } else {
          updated_lower = -std::numeric_limits<double>::infinity();
          updated_upper = shifted_rhs;
        }
      } else {
        if (!std::isfinite(state.col_upper[static_cast<std::size_t>(col)])) {
          continue;
        }
        const double shifted_rhs =
            rhs - coefficient * state.col_upper[static_cast<std::size_t>(col)];
        if (coefficient > 0.0) {
          updated_lower = shifted_rhs;
          updated_upper = std::numeric_limits<double>::infinity();
        } else {
          updated_lower = -std::numeric_limits<double>::infinity();
          updated_upper = shifted_rhs;
        }
      }

      state.set_row_bounds(row, updated_lower, updated_upper, options.bound_tolerance);
      state.remove_row_col_entry(row, col, nullptr);
      state.deactivate_empty_col(col);
    }
    ++reductions;
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

PresolveResult apply_singleton_cols_eq_mutable(const LpModel& model,
                                               const PresolveOptions& options) {
  MutablePresolveState state(model, options.zero_tolerance, options.bound_tolerance);
  MutableReductionOutcome outcome = process_singleton_cols_eq(state, options);
  if (outcome.status != PresolveStatus::kOk) {
    return PresolveResult(model, outcome.status, false, {}, 0.0);
  }
  if (outcome.reductions == 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }
  LpModel reduced = state.export_model(state.active_column_map());
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, state.objective_delta);
}

MutableReductionOutcome process_doubleton_eq(MutablePresolveState& state,
                                             const PresolveOptions& options) {
  if (options.doubleton_eq_batch_mode) {
    return process_doubleton_eq_batch(state, options);
  }

  state.seed_doubleton_rows(options.bound_tolerance);

  int reductions = 0;
  while (!state.doubleton_row_queue.empty()) {
    if (options.doubleton_eq_max_reductions >= 0 &&
        state.doubleton_eq_reductions_total >= options.doubleton_eq_max_reductions) {
      break;
    }
    const int row = state.doubleton_row_queue.front();
    state.doubleton_row_queue.pop_front();
    state.doubleton_row_queued[static_cast<std::size_t>(row)] = 0;

    if (!state.row_is_active_doubleton_eq(row, options.bound_tolerance)) {
      continue;
    }

    const MutableConstEntryRange row_entries = state.rows.entries(row);
    const int col1 = row_entries[0].index;
    const int col2 = row_entries[1].index;
    const double val1 = row_entries[0].value;
    const double val2 = row_entries[1].value;
    if (state.active_col[static_cast<std::size_t>(col1)] == 0 ||
        state.active_col[static_cast<std::size_t>(col2)] == 0) {
      continue;
    }

    int elim_col = -1;
    int keep_col = -1;
    double elim_val = 0.0;
    double keep_val = 0.0;
    choose_doubleton_columns(
        col1,
        val1,
        col2,
        val2,
        state.col_size,
        options.bound_tolerance,
        elim_col,
        elim_val,
        keep_col,
        keep_val);
    if (!acceptable_doubleton_pivot(keep_val, elim_val)) {
      continue;
    }

    const double rhs = state.row_upper[static_cast<std::size_t>(row)];
    const double alpha = -keep_val / elim_val;
    const double beta = rhs / elim_val;
    double mapped_lower = 0.0;
    double mapped_upper = 0.0;
    doubleton_mapped_interval(
        rhs,
        keep_val,
        elim_val,
        state.col_lower[static_cast<std::size_t>(elim_col)],
        state.col_upper[static_cast<std::size_t>(elim_col)],
        mapped_lower,
        mapped_upper);

    const double old_keep_lower = state.col_lower[static_cast<std::size_t>(keep_col)];
    const double old_keep_upper = state.col_upper[static_cast<std::size_t>(keep_col)];
    const double updated_lower = std::max(old_keep_lower, mapped_lower);
    const double updated_upper = std::min(old_keep_upper, mapped_upper);
    if (updated_lower > updated_upper + options.bound_tolerance) {
      return MutableReductionOutcome{PresolveStatus::kInfeasible, reductions, {}};
    }

    state.set_tighter_col_bounds(keep_col, updated_lower, updated_upper);

    const double elim_objective = state.objective[static_cast<std::size_t>(elim_col)];
    state.objective[static_cast<std::size_t>(keep_col)] += elim_objective * alpha;
    state.objective_delta += elim_objective * beta;

    state.remove_doubleton_pivot_row(row, elim_col, keep_col);
    while (!state.cols.empty(elim_col)) {
      const MutableEntry elim_entry =
          state.cols.back(elim_col);
      state.cols.pop_back(elim_col);
      --state.col_size[static_cast<std::size_t>(elim_col)];
      const int affected_row = elim_entry.index;
      if (state.active_row[static_cast<std::size_t>(affected_row)] == 0) {
        continue;
      }
      double elim_coeff = 0.0;
      if (!state.substitute_doubleton_row_entry(
              affected_row,
              elim_col,
              keep_col,
              alpha * elim_entry.value,
              &elim_coeff)) {
        continue;
      }

    state.shift_row_bounds_preserving_locks(affected_row, elim_coeff * beta, options.bound_tolerance);
    }

    state.touch_structure_changed();
    state.touch_parallel_signatures();
    state.deactivate_empty_col(elim_col);
    state.enqueue_singleton_dual_col(col1);
    state.enqueue_singleton_dual_col(col2);
    state.enqueue_singleton_dual_col(keep_col);
    state.enqueue_singleton_col(col1);
    state.enqueue_singleton_col(col2);
    state.enqueue_singleton_col(keep_col);
    ++state.doubleton_eq_reductions_total;
    ++reductions;
  }

  return MutableReductionOutcome{PresolveStatus::kOk, reductions, {}};
}

PresolveResult apply_doubleton_eq(const LpModel& model, const PresolveOptions& options) {
  MutablePresolveState state(model, options.zero_tolerance, options.bound_tolerance);
  MutableReductionOutcome outcome = process_doubleton_eq(state, options);
  if (outcome.status != PresolveStatus::kOk) {
    return PresolveResult(model, outcome.status, false, {}, 0.0);
  }
  if (outcome.reductions == 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  LpModel reduced = state.export_model(state.active_column_map());

  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, state.objective_delta);
}

bool zero_lower_two_col_row(const LpModel& model,
                            int row,
                            const PresolveOptions& options,
                            int& col1,
                            int& col2,
                            double& val1,
                            double& val2) {
  const double lower = model.row_lower()[row];
  const double upper = model.row_upper()[row];
  double scale = 0.0;
  if (std::isfinite(lower) && !std::isfinite(upper) &&
      std::fabs(lower) <= options.bound_tolerance) {
    scale = 1.0;
  } else if (!std::isfinite(lower) && std::isfinite(upper) &&
             std::fabs(upper) <= options.bound_tolerance) {
    scale = -1.0;
  } else {
    return false;
  }

  const CsrMatrix& csr = model.csr();
  int count = 0;
  col1 = -1;
  col2 = -1;
  val1 = 0.0;
  val2 = 0.0;
  for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
    const double value = scale * csr.values()[pos];
    if (std::fabs(value) <= options.zero_tolerance) {
      continue;
    }
    ++count;
    if (count == 1) {
      col1 = csr.col_idx()[pos];
      val1 = value;
    } else if (count == 2) {
      col2 = csr.col_idx()[pos];
      val2 = value;
    } else {
      return false;
    }
  }
  if (count != 2) {
    return false;
  }
  if (col2 < col1) {
    std::swap(col1, col2);
    std::swap(val1, val2);
  }
  return true;
}

bool extract_l1_split_pair(int row1_col1,
                           int row1_col2,
                           double row1_val1,
                           double row1_val2,
                           int row2_col1,
                           int row2_col2,
                           double row2_val1,
                           double row2_val2,
                           const PresolveOptions& options,
                           int& t_col,
                           int& e_col) {
  if (row1_col1 != row2_col1 || row1_col2 != row2_col2) {
    return false;
  }

  const int cols[2] = {row1_col1, row1_col2};
  const double row1_vals[2] = {row1_val1, row1_val2};
  const double row2_vals[2] = {row2_val1, row2_val2};
  for (int t_pos = 0; t_pos < 2; ++t_pos) {
    const int e_pos = 1 - t_pos;
    if (row1_vals[t_pos] <= options.zero_tolerance ||
        row2_vals[t_pos] <= options.zero_tolerance) {
      continue;
    }
    const double ratio1 = row1_vals[e_pos] / row1_vals[t_pos];
    const double ratio2 = row2_vals[e_pos] / row2_vals[t_pos];
    const double lo = std::min(ratio1, ratio2);
    const double hi = std::max(ratio1, ratio2);
    if (std::fabs(lo + 1.0) <= options.bound_tolerance &&
        std::fabs(hi - 1.0) <= options.bound_tolerance) {
      t_col = cols[t_pos];
      e_col = cols[e_pos];
      return true;
    }
  }
  return false;
}

bool row_contains_col(const CsrMatrix& csr, int row, int target_col, double zero_tolerance) {
  for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
    if (csr.col_idx()[pos] == target_col && std::fabs(csr.values()[pos]) > zero_tolerance) {
      return true;
    }
  }
  return false;
}

struct GraphL1Block {
  int coupling_row;
  int epi_row_a;
  int epi_row_b;
  int t_col;
  int e_col;
  double rho;
};

bool try_graph_l1_orientation(const LpModel& model,
                              int col1,
                              double row_a_val1,
                              double row_a_val2,
                              double row_b_val1,
                              double row_b_val2,
                              int candidate_t,
                              int candidate_e,
                              const PresolveOptions& options,
                              double& rho) {
  if (!std::isinf(model.col_lower()[candidate_e]) ||
      !std::isinf(model.col_upper()[candidate_e]) ||
      std::fabs(model.col_lower()[candidate_t]) > options.bound_tolerance ||
      !std::isinf(model.col_upper()[candidate_t]) ||
      model.objective()[candidate_t] < -options.bound_tolerance) {
    return false;
  }

  const double vals_a[2] = {row_a_val1, row_a_val2};
  const double vals_b[2] = {row_b_val1, row_b_val2};
  const int t_pos = candidate_t == col1 ? 0 : 1;
  const int e_pos = candidate_e == col1 ? 0 : 1;
  if (t_pos == e_pos ||
      vals_a[t_pos] <= options.zero_tolerance ||
      vals_b[t_pos] <= options.zero_tolerance) {
    return false;
  }

  const double rho_a = vals_a[e_pos] / vals_a[t_pos];
  const double rho_b = vals_b[e_pos] / vals_b[t_pos];
  const double scale = std::max({1.0, std::fabs(rho_a), std::fabs(rho_b)});
  if (std::fabs(rho_a + rho_b) > options.bound_tolerance * scale ||
      std::fabs(rho_a) <= options.zero_tolerance) {
    return false;
  }
  rho = std::fabs(rho_a);
  return true;
}

bool find_graph_l1_blocks(const LpModel& model,
                          const PresolveOptions& options,
                          std::vector<GraphL1Block>& blocks,
                          std::vector<unsigned char>& removable_rows,
                          std::vector<unsigned char>& removable_cols) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();

  std::map<std::pair<int, int>, std::vector<int>> row_groups;
  std::vector<int> z_col1(static_cast<std::size_t>(m), -1);
  std::vector<int> z_col2(static_cast<std::size_t>(m), -1);
  std::vector<double> z_val1(static_cast<std::size_t>(m), 0.0);
  std::vector<double> z_val2(static_cast<std::size_t>(m), 0.0);

  for (int row = 0; row < m; ++row) {
    int col1 = -1;
    int col2 = -1;
    double val1 = 0.0;
    double val2 = 0.0;
    if (!zero_lower_two_col_row(model, row, options, col1, col2, val1, val2)) {
      continue;
    }
    z_col1[static_cast<std::size_t>(row)] = col1;
    z_col2[static_cast<std::size_t>(row)] = col2;
    z_val1[static_cast<std::size_t>(row)] = val1;
    z_val2[static_cast<std::size_t>(row)] = val2;
    row_groups[{col1, col2}].push_back(row);
  }

  std::vector<unsigned char> used_t(static_cast<std::size_t>(n), 0);
  std::vector<unsigned char> used_e(static_cast<std::size_t>(n), 0);
  std::vector<GraphL1Block> candidates;
  std::vector<std::pair<int, int>> epi_rows;

  for (const auto& group : row_groups) {
    const std::vector<int>& rows = group.second;
    if (rows.size() < 2) {
      continue;
    }
    const int col1 = group.first.first;
    const int col2 = group.first.second;
    bool group_claimed = false;
    for (std::size_t a = 0; a + 1 < rows.size() && !group_claimed; ++a) {
      for (std::size_t b = a + 1; b < rows.size() && !group_claimed; ++b) {
        const int row_a = rows[a];
        const int row_b = rows[b];
        int t_col = -1;
        int e_col = -1;
        double rho = 0.0;
        if (try_graph_l1_orientation(
                model,
                col1,
                z_val1[static_cast<std::size_t>(row_a)],
                z_val2[static_cast<std::size_t>(row_a)],
                z_val1[static_cast<std::size_t>(row_b)],
                z_val2[static_cast<std::size_t>(row_b)],
                col1,
                col2,
                options,
                rho)) {
          t_col = col1;
          e_col = col2;
        } else if (try_graph_l1_orientation(
                       model,
                       col1,
                       z_val1[static_cast<std::size_t>(row_a)],
                       z_val2[static_cast<std::size_t>(row_a)],
                       z_val1[static_cast<std::size_t>(row_b)],
                       z_val2[static_cast<std::size_t>(row_b)],
                       col2,
                       col1,
                       options,
                       rho)) {
          t_col = col2;
          e_col = col1;
        } else {
          continue;
        }

        if (used_t[static_cast<std::size_t>(t_col)] != 0 ||
            used_e[static_cast<std::size_t>(e_col)] != 0) {
          continue;
        }

        std::vector<int> coupling_rows;
        for (int pos = csc.col_ptr()[e_col]; pos < csc.col_ptr()[e_col + 1]; ++pos) {
          const int row = csc.row_idx()[pos];
          if (row == row_a || row == row_b ||
              std::fabs(csc.values()[pos]) <= options.zero_tolerance) {
            continue;
          }
          coupling_rows.push_back(row);
        }
        if (coupling_rows.size() != 1 ||
            !is_equality_row(
                model.row_lower()[coupling_rows[0]],
                model.row_upper()[coupling_rows[0]],
                options.bound_tolerance)) {
          continue;
        }

        used_t[static_cast<std::size_t>(t_col)] = 1;
        used_e[static_cast<std::size_t>(e_col)] = 1;
        candidates.push_back(GraphL1Block{
            coupling_rows[0],
            row_a,
            row_b,
            t_col,
            e_col,
            rho});
        epi_rows.push_back({row_a, row_b});
        group_claimed = true;
      }
    }
  }

  if (candidates.empty()) {
    return false;
  }

  std::vector<unsigned char> epi_mask(static_cast<std::size_t>(m), 0);
  for (const auto& rows : epi_rows) {
    epi_mask[static_cast<std::size_t>(rows.first)] = 1;
    epi_mask[static_cast<std::size_t>(rows.second)] = 1;
  }

  std::map<int, std::vector<int>> slack_rows;
  std::map<int, std::vector<int>> block_extra_rows;
  std::vector<unsigned char> block_bad(candidates.size(), 0);

  for (std::size_t block_index = 0; block_index < candidates.size(); ++block_index) {
    const GraphL1Block& block = candidates[block_index];
    for (int pos = csc.col_ptr()[block.t_col]; pos < csc.col_ptr()[block.t_col + 1]; ++pos) {
      const int row = csc.row_idx()[pos];
      if (epi_mask[static_cast<std::size_t>(row)] != 0 ||
          std::fabs(csc.values()[pos]) <= options.zero_tolerance) {
        continue;
      }

      int col1 = -1;
      int col2 = -1;
      double val1 = 0.0;
      double val2 = 0.0;
      if (!zero_lower_two_col_row(model, row, options, col1, col2, val1, val2) ||
          (col1 != block.t_col && col2 != block.t_col)) {
        block_bad[block_index] = 1;
        continue;
      }
      const double coeff_t = col1 == block.t_col ? val1 : val2;
      const int slack_col = col1 == block.t_col ? col2 : col1;
      const double coeff_s = col1 == block.t_col ? val2 : val1;
      if (coeff_t >= -options.zero_tolerance || coeff_s <= options.zero_tolerance) {
        block_bad[block_index] = 1;
        continue;
      }
      slack_rows[slack_col].push_back(row);
      block_extra_rows[static_cast<int>(block_index)].push_back(row);
    }
  }

  std::set<int> removable_slack_cols;
  std::set<int> removable_slack_rows;
  for (const auto& entry : slack_rows) {
    const int slack_col = entry.first;
    if (std::fabs(model.objective()[slack_col]) > options.zero_tolerance ||
        !std::isinf(model.col_upper()[slack_col])) {
      continue;
    }

    std::set<int> all_slack_rows;
    for (int pos = csc.col_ptr()[slack_col]; pos < csc.col_ptr()[slack_col + 1]; ++pos) {
      if (std::fabs(csc.values()[pos]) > options.zero_tolerance) {
        all_slack_rows.insert(csc.row_idx()[pos]);
      }
    }
    std::set<int> candidate_rows(entry.second.begin(), entry.second.end());
    if (all_slack_rows != candidate_rows) {
      continue;
    }
    removable_slack_cols.insert(slack_col);
    removable_slack_rows.insert(candidate_rows.begin(), candidate_rows.end());
  }

  std::vector<GraphL1Block> valid_blocks;
  for (std::size_t block_index = 0; block_index < candidates.size(); ++block_index) {
    if (block_bad[block_index] != 0) {
      continue;
    }
    bool extras_ok = true;
    for (int row : block_extra_rows[static_cast<int>(block_index)]) {
      if (removable_slack_rows.find(row) == removable_slack_rows.end()) {
        extras_ok = false;
        break;
      }
    }
    if (!extras_ok) {
      continue;
    }
    valid_blocks.push_back(candidates[block_index]);
  }

  if (valid_blocks.empty()) {
    return false;
  }

  blocks = std::move(valid_blocks);
  removable_rows.assign(static_cast<std::size_t>(m), 0);
  removable_cols.assign(static_cast<std::size_t>(n), 0);
  for (const GraphL1Block& block : blocks) {
    removable_rows[static_cast<std::size_t>(block.epi_row_a)] = 1;
    removable_rows[static_cast<std::size_t>(block.epi_row_b)] = 1;
  }
  for (int row : removable_slack_rows) {
    removable_rows[static_cast<std::size_t>(row)] = 1;
  }
  for (int col : removable_slack_cols) {
    removable_cols[static_cast<std::size_t>(col)] = 1;
  }
  return true;
}

PresolveResult apply_structural_l1_graph_substitution(const LpModel& model,
                                                      const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  std::vector<GraphL1Block> blocks;
  std::vector<unsigned char> remove_row;
  std::vector<unsigned char> remove_col;
  if (!find_graph_l1_blocks(model, options, blocks, remove_row, remove_col)) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::map<int, GraphL1Block> block_by_e;
  std::map<int, GraphL1Block> block_by_t;
  std::vector<double> new_objective = model.objective();
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();
  for (const GraphL1Block& block : blocks) {
    block_by_e[block.e_col] = block;
    block_by_t[block.t_col] = block;
    const double inv_rho = 1.0 / block.rho;
    new_objective[static_cast<std::size_t>(block.t_col)] =
        model.objective()[block.t_col] + model.objective()[block.e_col] * inv_rho;
    new_objective[static_cast<std::size_t>(block.e_col)] =
        model.objective()[block.t_col] - model.objective()[block.e_col] * inv_rho;
    new_col_lower[static_cast<std::size_t>(block.t_col)] = 0.0;
    new_col_lower[static_cast<std::size_t>(block.e_col)] = 0.0;
    new_col_upper[static_cast<std::size_t>(block.t_col)] = std::numeric_limits<double>::infinity();
    new_col_upper[static_cast<std::size_t>(block.e_col)] = std::numeric_limits<double>::infinity();
  }

  std::vector<int> old_to_new_row(static_cast<std::size_t>(m), -1);
  std::vector<int> old_to_new_col(static_cast<std::size_t>(n), -1);
  std::vector<double> kept_row_lower;
  std::vector<double> kept_row_upper;
  std::vector<double> kept_col_lower;
  std::vector<double> kept_col_upper;
  std::vector<double> kept_objective;

  int next_row = 0;
  for (int row = 0; row < m; ++row) {
    if (remove_row[static_cast<std::size_t>(row)] != 0) {
      continue;
    }
    old_to_new_row[static_cast<std::size_t>(row)] = next_row++;
    kept_row_lower.push_back(model.row_lower()[row]);
    kept_row_upper.push_back(model.row_upper()[row]);
  }

  int next_col = 0;
  for (int col = 0; col < n; ++col) {
    if (remove_col[static_cast<std::size_t>(col)] != 0) {
      continue;
    }
    old_to_new_col[static_cast<std::size_t>(col)] = next_col++;
    kept_objective.push_back(new_objective[static_cast<std::size_t>(col)]);
    kept_col_lower.push_back(new_col_lower[static_cast<std::size_t>(col)]);
    kept_col_upper.push_back(new_col_upper[static_cast<std::size_t>(col)]);
  }

  const CsrMatrix& csr = model.csr();
  std::vector<RowEntries> row_entries(static_cast<std::size_t>(next_row));
  for (int row = 0; row < m; ++row) {
    const int new_row = old_to_new_row[static_cast<std::size_t>(row)];
    if (new_row < 0) {
      continue;
    }
    RowEntries& entries = row_entries[static_cast<std::size_t>(new_row)];
    for (int pos = csr.row_ptr()[row]; pos < csr.row_ptr()[row + 1]; ++pos) {
      const int col = csr.col_idx()[pos];
      const double value = csr.values()[pos];
      const auto e_block = block_by_e.find(col);
      const auto t_block = block_by_t.find(col);
      if (e_block != block_by_e.end()) {
        const GraphL1Block& block = e_block->second;
        add_row_entry(
            entries,
            old_to_new_col[static_cast<std::size_t>(block.t_col)],
            value / block.rho,
            options.zero_tolerance);
        add_row_entry(
            entries,
            old_to_new_col[static_cast<std::size_t>(block.e_col)],
            -value / block.rho,
            options.zero_tolerance);
      } else if (t_block != block_by_t.end()) {
        const GraphL1Block& block = t_block->second;
        add_row_entry(
            entries,
            old_to_new_col[static_cast<std::size_t>(block.t_col)],
            value,
            options.zero_tolerance);
        add_row_entry(
            entries,
            old_to_new_col[static_cast<std::size_t>(block.e_col)],
            value,
            options.zero_tolerance);
      } else {
        const int new_col = old_to_new_col[static_cast<std::size_t>(col)];
        if (new_col < 0) {
          continue;
        }
        add_row_entry(entries, new_col, value, options.zero_tolerance);
      }
    }
  }

  LpModel reduced = make_model_from_row_entries(
      next_row,
      next_col,
      row_entries,
      std::move(kept_objective),
      std::move(kept_row_lower),
      std::move(kept_row_upper),
      std::move(kept_col_lower),
      std::move(kept_col_upper),
      model.obj_constant(),
      options.zero_tolerance);
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

PresolveResult apply_structural_l1_substitution(const LpModel& model,
                                                const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  PresolveResult graph_result = apply_structural_l1_graph_substitution(model, options);
  if (graph_result.changed()) {
    return graph_result;
  }

  if (m == 0 || m % 3 != 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  const int blocks = m / 3;
  const CsrMatrix& csr = model.csr();
  std::vector<int> t_cols(static_cast<std::size_t>(blocks), -1);
  std::vector<int> e_cols(static_cast<std::size_t>(blocks), -1);
  std::vector<unsigned char> used_col(static_cast<std::size_t>(n), 0);

  for (int block = 0; block < blocks; ++block) {
    const int eq_row = 3 * block;
    const int row1 = eq_row + 1;
    const int row2 = eq_row + 2;
    if (!is_equality_row(
            model.row_lower()[eq_row], model.row_upper()[eq_row], options.bound_tolerance)) {
      return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
    }

    int row1_col1 = -1;
    int row1_col2 = -1;
    int row2_col1 = -1;
    int row2_col2 = -1;
    double row1_val1 = 0.0;
    double row1_val2 = 0.0;
    double row2_val1 = 0.0;
    double row2_val2 = 0.0;
    if (!zero_lower_two_col_row(
            model, row1, options, row1_col1, row1_col2, row1_val1, row1_val2) ||
        !zero_lower_two_col_row(
            model, row2, options, row2_col1, row2_col2, row2_val1, row2_val2)) {
      return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
    }

    int t_col = -1;
    int e_col = -1;
    if (!extract_l1_split_pair(
            row1_col1,
            row1_col2,
            row1_val1,
            row1_val2,
            row2_col1,
            row2_col2,
            row2_val1,
            row2_val2,
            options,
            t_col,
            e_col)) {
      return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
    }

    if (used_col[static_cast<std::size_t>(t_col)] != 0 ||
        used_col[static_cast<std::size_t>(e_col)] != 0 ||
        std::fabs(model.col_lower()[t_col]) > options.bound_tolerance ||
        !std::isinf(model.col_upper()[t_col]) ||
        !std::isinf(model.col_lower()[e_col]) ||
        !std::isinf(model.col_upper()[e_col]) ||
        !row_contains_col(csr, eq_row, e_col, options.zero_tolerance)) {
      return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
    }

    used_col[static_cast<std::size_t>(t_col)] = 1;
    used_col[static_cast<std::size_t>(e_col)] = 1;
    t_cols[static_cast<std::size_t>(block)] = t_col;
    e_cols[static_cast<std::size_t>(block)] = e_col;
  }

  std::vector<double> new_objective = model.objective();
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();
  for (int block = 0; block < blocks; ++block) {
    const int t_col = t_cols[static_cast<std::size_t>(block)];
    const int e_col = e_cols[static_cast<std::size_t>(block)];
    new_objective[static_cast<std::size_t>(t_col)] =
        model.objective()[t_col] + model.objective()[e_col];
    new_objective[static_cast<std::size_t>(e_col)] =
        model.objective()[t_col] - model.objective()[e_col];
    new_col_lower[static_cast<std::size_t>(t_col)] = 0.0;
    new_col_lower[static_cast<std::size_t>(e_col)] = 0.0;
    new_col_upper[static_cast<std::size_t>(t_col)] = std::numeric_limits<double>::infinity();
    new_col_upper[static_cast<std::size_t>(e_col)] = std::numeric_limits<double>::infinity();
  }

  std::vector<RowEntries> row_entries(static_cast<std::size_t>(blocks));
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  row_lower.reserve(static_cast<std::size_t>(blocks));
  row_upper.reserve(static_cast<std::size_t>(blocks));
  for (int block = 0; block < blocks; ++block) {
    const int old_row = 3 * block;
    row_lower.push_back(model.row_lower()[old_row]);
    row_upper.push_back(model.row_upper()[old_row]);
    const int t_col = t_cols[static_cast<std::size_t>(block)];
    const int e_col = e_cols[static_cast<std::size_t>(block)];
    RowEntries& entries = row_entries[static_cast<std::size_t>(block)];
    for (int pos = csr.row_ptr()[old_row]; pos < csr.row_ptr()[old_row + 1]; ++pos) {
      const int col = csr.col_idx()[pos];
      const double value = csr.values()[pos];
      if (col == e_col) {
        add_row_entry(entries, t_col, value, options.zero_tolerance);
        add_row_entry(entries, e_col, -value, options.zero_tolerance);
      } else {
        add_row_entry(entries, col, value, options.zero_tolerance);
      }
    }
  }

  LpModel reduced = make_model_from_row_entries(
      blocks,
      n,
      row_entries,
      std::move(new_objective),
      std::move(row_lower),
      std::move(row_upper),
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant(),
      options.zero_tolerance);
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

PresolveResult apply_dual_fix(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();
  std::vector<unsigned char> fixed(static_cast<std::size_t>(n), 0);
  std::vector<double> row_shift(static_cast<std::size_t>(m), 0.0);
  std::vector<FixedColumnRecord> fixed_columns;
  double objective_shift = 0.0;

  for (int col = 0; col < n; ++col) {
    bool has_down_lock = false;
    bool has_up_lock = false;
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int row = csc.row_idx()[pos];
      const double a = csc.values()[pos];
      if (a > options.zero_tolerance) {
        has_down_lock = has_down_lock || std::isfinite(model.row_lower()[row]);
        has_up_lock = has_up_lock || std::isfinite(model.row_upper()[row]);
      } else if (a < -options.zero_tolerance) {
        has_down_lock = has_down_lock || std::isfinite(model.row_upper()[row]);
        has_up_lock = has_up_lock || std::isfinite(model.row_lower()[row]);
      }
      if (has_down_lock && has_up_lock) {
        break;
      }
    }

    const double objective = model.objective()[col];
    const double lower = model.col_lower()[col];
    const double upper = model.col_upper()[col];
    bool should_fix = false;
    double value = 0.0;

    if (objective > options.zero_tolerance && !has_down_lock) {
      if (!std::isfinite(lower)) {
        return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
      }
      should_fix = true;
      value = lower;
    } else if (objective < -options.zero_tolerance && !has_up_lock) {
      if (!std::isfinite(upper)) {
        return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
      }
      should_fix = true;
      value = upper;
    } else if (std::fabs(objective) <= options.zero_tolerance) {
      if (!has_down_lock && std::isfinite(lower)) {
        should_fix = true;
        value = lower;
      } else if (!has_up_lock && std::isfinite(upper)) {
        should_fix = true;
        value = upper;
      }
    }

    if (!should_fix) {
      continue;
    }

    fixed[static_cast<std::size_t>(col)] = 1;
    objective_shift += objective * value;
    fixed_columns.push_back(FixedColumnRecord{col, value, objective});
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      const int row = csc.row_idx()[pos];
      row_shift[static_cast<std::size_t>(row)] += csc.values()[pos] * value;
    }
  }

  if (fixed_columns.empty()) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  for (int row = 0; row < m; ++row) {
    new_row_lower[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
    new_row_upper[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
  }

  int kept_cols = 0;
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  std::vector<double> new_objective;
  std::vector<double> new_col_lower;
  std::vector<double> new_col_upper;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    if (fixed[static_cast<std::size_t>(col)] != 0) {
      continue;
    }
    ++kept_cols;
    new_objective.push_back(model.objective()[col]);
    new_col_lower.push_back(model.col_lower()[col]);
    new_col_upper.push_back(model.col_upper()[col]);
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      new_row_idx.push_back(csc.row_idx()[pos]);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m, kept_cols, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      std::move(new_objective),
      std::move(new_row_lower),
      std::move(new_row_upper),
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant() + objective_shift);

  return PresolveResult(
      std::move(reduced),
      PresolveStatus::kOk,
      true,
      std::move(fixed_columns),
      objective_shift);
}

bool empty_col_is_unbounded(double objective,
                            double lower,
                            double upper,
                            double zero_tolerance) {
  return (objective > zero_tolerance && !std::isfinite(lower)) ||
         (objective < -zero_tolerance && !std::isfinite(upper));
}

double empty_col_fixed_value(double objective,
                             double lower,
                             double upper,
                             double zero_tolerance) {
  if (objective > zero_tolerance) {
    return lower;
  }
  if (objective < -zero_tolerance) {
    return upper;
  }
  if (std::isfinite(lower)) {
    return lower;
  }
  if (std::isfinite(upper)) {
    return upper;
  }
  return 0.0;
}

PresolveResult apply_empty_cols(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();

  std::vector<unsigned char> fixed(static_cast<std::size_t>(n), 0);
  std::vector<FixedColumnRecord> fixed_columns;
  double objective_shift = 0.0;

  for (int col = 0; col < n; ++col) {
    const bool is_empty = csc.col_ptr()[col] == csc.col_ptr()[col + 1];
    if (!is_empty) {
      continue;
    }

    const double objective = model.objective()[col];
    const double lower = model.col_lower()[col];
    const double upper = model.col_upper()[col];
    if (empty_col_is_unbounded(objective, lower, upper, options.zero_tolerance)) {
      return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
    }

    const double value = empty_col_fixed_value(objective, lower, upper, options.zero_tolerance);
    fixed[static_cast<std::size_t>(col)] = 1;
    fixed_columns.push_back(FixedColumnRecord{col, value, objective});
    objective_shift += objective * value;
  }

  if (fixed_columns.empty()) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  int kept_cols = 0;
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  std::vector<double> new_objective;
  std::vector<double> new_col_lower;
  std::vector<double> new_col_upper;
  new_col_ptr.reserve(static_cast<std::size_t>(n + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    if (fixed[static_cast<std::size_t>(col)] != 0) {
      continue;
    }

    ++kept_cols;
    new_objective.push_back(model.objective()[col]);
    new_col_lower.push_back(model.col_lower()[col]);
    new_col_upper.push_back(model.col_upper()[col]);
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      new_row_idx.push_back(csc.row_idx()[pos]);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m, kept_cols, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      std::move(new_objective),
      model.row_lower(),
      model.row_upper(),
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant() + objective_shift);

  return PresolveResult(
      std::move(reduced),
      PresolveStatus::kOk,
      true,
      std::move(fixed_columns),
      objective_shift);
}

bool redundant_upper_from_row(const LpModel& model,
                              const RowActivitySums& row_activity,
                              int row,
                              int target_col,
                              double target_a,
                              const PresolveOptions& options,
                              double current_upper) {
  (void)target_col;
  if (target_a > options.zero_tolerance && std::isfinite(model.row_upper()[row])) {
    double rest_min = 0.0;
    const double current_min = term_min(target_a, model.col_lower()[target_col], model.col_upper()[target_col]);
    if (!finite_activity_excluding(row_activity.minimum[static_cast<std::size_t>(row)], current_min, rest_min)) {
      return false;
    }
    const double implied_upper = (model.row_upper()[row] - rest_min) / target_a;
    return implied_upper <= current_upper + options.bound_tolerance;
  }

  if (target_a < -options.zero_tolerance && std::isfinite(model.row_lower()[row])) {
    double rest_max = 0.0;
    const double current_max = term_max(target_a, model.col_lower()[target_col], model.col_upper()[target_col]);
    if (!finite_activity_excluding(row_activity.maximum[static_cast<std::size_t>(row)], current_max, rest_max)) {
      return false;
    }
    const double implied_upper = (model.row_lower()[row] - rest_max) / target_a;
    return implied_upper <= current_upper + options.bound_tolerance;
  }

  return false;
}

bool redundant_lower_from_row(const LpModel& model,
                              const RowActivitySums& row_activity,
                              int row,
                              int target_col,
                              double target_a,
                              const PresolveOptions& options,
                              double current_lower) {
  (void)target_col;
  if (target_a > options.zero_tolerance && std::isfinite(model.row_lower()[row])) {
    double rest_max = 0.0;
    const double current_max = term_max(target_a, model.col_lower()[target_col], model.col_upper()[target_col]);
    if (!finite_activity_excluding(row_activity.maximum[static_cast<std::size_t>(row)], current_max, rest_max)) {
      return false;
    }
    const double implied_lower = (model.row_lower()[row] - rest_max) / target_a;
    return implied_lower >= current_lower - options.bound_tolerance;
  }

  if (target_a < -options.zero_tolerance && std::isfinite(model.row_upper()[row])) {
    double rest_min = 0.0;
    const double current_min = term_min(target_a, model.col_lower()[target_col], model.col_upper()[target_col]);
    if (!finite_activity_excluding(row_activity.minimum[static_cast<std::size_t>(row)], current_min, rest_min)) {
      return false;
    }
    const double implied_lower = (model.row_upper()[row] - rest_min) / target_a;
    return implied_lower >= current_lower - options.bound_tolerance;
  }

  return false;
}

PresolveResult apply_redundant_bounds(const LpModel& model, const PresolveOptions& options) {
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();
  const RowActivitySums row_activity = compute_row_activity_sums(model);
  bool changed = false;

  for (int col = 0; col < n; ++col) {
    const double lower = model.col_lower()[col];
    const double upper = model.col_upper()[col];
    if (!std::isfinite(lower) && std::isfinite(upper)) {
      for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
        if (redundant_upper_from_row(
                model,
                row_activity,
                csc.row_idx()[pos],
                col,
                csc.values()[pos],
                options,
                upper)) {
          new_col_upper[static_cast<std::size_t>(col)] = std::numeric_limits<double>::infinity();
          changed = true;
          break;
        }
      }
    } else if (std::isfinite(lower) && !std::isfinite(upper)) {
      for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
        if (redundant_lower_from_row(
                model,
                row_activity,
                csc.row_idx()[pos],
                col,
                csc.values()[pos],
                options,
                lower)) {
          new_col_lower[static_cast<std::size_t>(col)] = -std::numeric_limits<double>::infinity();
          changed = true;
          break;
        }
      }
    }
  }

  if (!changed) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  LpModel reduced = model.with_col_bounds(
      std::move(new_col_lower),
      std::move(new_col_upper),
      model.obj_constant());
  return PresolveResult(std::move(reduced), PresolveStatus::kOk, true, {}, 0.0);
}

bool parallel_col_ratio(const CscMatrix& csc,
                        int target_col,
                        int source_col,
                        double tolerance,
                        double& ratio) {
  const int target_start = csc.col_ptr()[target_col];
  const int target_stop = csc.col_ptr()[target_col + 1];
  const int source_start = csc.col_ptr()[source_col];
  const int source_stop = csc.col_ptr()[source_col + 1];
  const int len = target_stop - target_start;
  if (len <= 0 || source_stop - source_start != len) {
    return false;
  }

  bool ratio_set = false;
  ratio = 0.0;
  for (int offset = 0; offset < len; ++offset) {
    const int target_pos = target_start + offset;
    const int source_pos = source_start + offset;
    if (csc.row_idx()[target_pos] != csc.row_idx()[source_pos]) {
      return false;
    }
    const double target_value = csc.values()[target_pos];
    const double source_value = csc.values()[source_pos];
    if (std::fabs(target_value) <= tolerance) {
      return false;
    }
    if (!ratio_set) {
      ratio = source_value / target_value;
      ratio_set = true;
    }
    if (std::fabs(source_value - ratio * target_value) > tolerance) {
      return false;
    }
  }
  return ratio_set && std::fabs(ratio) > tolerance && std::isfinite(ratio);
}

double merged_parallel_col_lower(double target_lower,
                                 double source_lower,
                                 double source_upper,
                                 double ratio) {
  if (ratio > 0.0) {
    return std::isfinite(target_lower) && std::isfinite(source_lower)
               ? target_lower + ratio * source_lower
               : -std::numeric_limits<double>::infinity();
  }
  return std::isfinite(target_lower) && std::isfinite(source_upper)
             ? target_lower + ratio * source_upper
             : -std::numeric_limits<double>::infinity();
}

double merged_parallel_col_upper(double target_upper,
                                 double source_lower,
                                 double source_upper,
                                 double ratio) {
  if (ratio > 0.0) {
    return std::isfinite(target_upper) && std::isfinite(source_upper)
               ? target_upper + ratio * source_upper
               : std::numeric_limits<double>::infinity();
  }
  return std::isfinite(target_upper) && std::isfinite(source_lower)
             ? target_upper + ratio * source_lower
             : std::numeric_limits<double>::infinity();
}

PresolveResult apply_parallel_cols(const LpModel& model, const PresolveOptions& options) {
  const int m = model.num_rows();
  const int n = model.num_cols();
  const CscMatrix& csc = model.csc();
  std::vector<unsigned char> remove_col(static_cast<std::size_t>(n), 0);
  std::vector<unsigned char> fixed_col(static_cast<std::size_t>(n), 0);
  std::vector<double> new_col_lower = model.col_lower();
  std::vector<double> new_col_upper = model.col_upper();
  std::vector<double> row_shift(static_cast<std::size_t>(m), 0.0);
  std::vector<FixedColumnRecord> fixed_columns;
  double objective_delta = 0.0;
  int removed_cols = 0;

  const std::vector<SparsePatternSignature> signatures = col_pattern_signatures(csc);
  for (std::size_t group_begin = 0; group_begin < signatures.size();) {
    std::size_t group_end = group_begin + 1;
    while (group_end < signatures.size() &&
           signatures[group_end].length == signatures[group_begin].length &&
           signatures[group_end].hash == signatures[group_begin].hash &&
           signatures[group_end].value_hash == signatures[group_begin].value_hash) {
      ++group_end;
    }

    if (group_end - group_begin < 2) {
      group_begin = group_end;
      continue;
    }

    for (std::size_t target_pos = group_begin; target_pos + 1 < group_end; ++target_pos) {
      const int target = signatures[target_pos].index;
      if (remove_col[static_cast<std::size_t>(target)] != 0) {
        continue;
      }

      for (std::size_t source_pos = target_pos + 1; source_pos < group_end; ++source_pos) {
        const int source = signatures[source_pos].index;
        if (remove_col[static_cast<std::size_t>(source)] != 0) {
          continue;
        }
        double ratio = 0.0;
        if (!parallel_col_ratio(csc, target, source, options.zero_tolerance, ratio)) {
          continue;
        }

        const double obj_gap = model.objective()[source] - ratio * model.objective()[target];
        if (std::fabs(obj_gap) <= options.zero_tolerance) {
          const double target_lower = new_col_lower[static_cast<std::size_t>(target)];
          const double target_upper = new_col_upper[static_cast<std::size_t>(target)];
          new_col_lower[static_cast<std::size_t>(target)] = merged_parallel_col_lower(
              target_lower,
              model.col_lower()[source],
              model.col_upper()[source],
              ratio);
          new_col_upper[static_cast<std::size_t>(target)] = merged_parallel_col_upper(
              target_upper,
              model.col_lower()[source],
              model.col_upper()[source],
              ratio);
          remove_col[static_cast<std::size_t>(source)] = 1;
          ++removed_cols;
          continue;
        }

        bool fix_source_to_lower = false;
        bool fix_source_to_upper = false;
        bool fix_target_to_lower = false;
        bool fix_target_to_upper = false;
        const double target_lower = new_col_lower[static_cast<std::size_t>(target)];
        const double target_upper = new_col_upper[static_cast<std::size_t>(target)];
        const double source_lower = model.col_lower()[source];
        const double source_upper = model.col_upper()[source];

        if (obj_gap > options.zero_tolerance) {
          if (ratio > 0.0) {
            fix_source_to_lower = !std::isfinite(target_upper);
            fix_target_to_upper = !std::isfinite(source_lower);
          } else {
            fix_source_to_lower = !std::isfinite(target_lower);
            fix_target_to_lower = !std::isfinite(source_lower);
          }
        } else {
          if (ratio > 0.0) {
            fix_source_to_upper = !std::isfinite(target_lower);
            fix_target_to_lower = !std::isfinite(source_upper);
          } else {
            fix_source_to_upper = !std::isfinite(target_upper);
            fix_target_to_upper = !std::isfinite(source_upper);
          }
        }

        auto fix_column = [&](int col, double value) {
          fixed_col[static_cast<std::size_t>(col)] = 1;
          remove_col[static_cast<std::size_t>(col)] = 1;
          ++removed_cols;
          objective_delta += model.objective()[col] * value;
          fixed_columns.push_back(FixedColumnRecord{col, value, model.objective()[col]});
          for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
            const int row = csc.row_idx()[pos];
            row_shift[static_cast<std::size_t>(row)] += csc.values()[pos] * value;
          }
        };

        if (fix_source_to_lower) {
          if (!std::isfinite(source_lower)) {
            return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
          }
          fix_column(source, source_lower);
          continue;
        }
        if (fix_source_to_upper) {
          if (!std::isfinite(source_upper)) {
            return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
          }
          fix_column(source, source_upper);
          continue;
        }
        if (fix_target_to_lower) {
          if (!std::isfinite(target_lower)) {
            return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
          }
          fix_column(target, target_lower);
          break;
        }
        if (fix_target_to_upper) {
          if (!std::isfinite(target_upper)) {
            return PresolveResult(model, PresolveStatus::kUnbounded, false, {}, 0.0);
          }
          fix_column(target, target_upper);
          break;
        }
      }
    }
    group_begin = group_end;
  }

  if (removed_cols == 0) {
    return PresolveResult(model, PresolveStatus::kOk, false, {}, 0.0);
  }

  std::vector<double> new_row_lower = model.row_lower();
  std::vector<double> new_row_upper = model.row_upper();
  for (int row = 0; row < m; ++row) {
    new_row_lower[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
    new_row_upper[static_cast<std::size_t>(row)] -= row_shift[static_cast<std::size_t>(row)];
  }

  int kept_cols = 0;
  std::vector<int> new_col_ptr;
  std::vector<int> new_row_idx;
  std::vector<double> new_values;
  std::vector<double> new_objective;
  std::vector<double> kept_col_lower;
  std::vector<double> kept_col_upper;
  new_col_ptr.reserve(static_cast<std::size_t>(n - removed_cols + 1));
  new_col_ptr.push_back(0);

  for (int col = 0; col < n; ++col) {
    if (remove_col[static_cast<std::size_t>(col)] != 0) {
      continue;
    }
    ++kept_cols;
    new_objective.push_back(model.objective()[col]);
    kept_col_lower.push_back(new_col_lower[static_cast<std::size_t>(col)]);
    kept_col_upper.push_back(new_col_upper[static_cast<std::size_t>(col)]);
    for (int pos = csc.col_ptr()[col]; pos < csc.col_ptr()[col + 1]; ++pos) {
      new_row_idx.push_back(csc.row_idx()[pos]);
      new_values.push_back(csc.values()[pos]);
    }
    new_col_ptr.push_back(static_cast<int>(new_values.size()));
  }

  LpModel reduced(
      CscMatrix(m, kept_cols, std::move(new_col_ptr), std::move(new_row_idx), std::move(new_values)),
      std::move(new_objective),
      std::move(new_row_lower),
      std::move(new_row_upper),
      std::move(kept_col_lower),
      std::move(kept_col_upper),
      model.obj_constant() + objective_delta);
  return PresolveResult(
      std::move(reduced),
      PresolveStatus::kOk,
      true,
      std::move(fixed_columns),
      objective_delta);
}

struct SchedulerState {
  explicit SchedulerState(const LpModel& initial_model) : current(initial_model) {}

  LpModel current;
  std::unique_ptr<MutablePresolveState> mutable_state;
  bool changed = false;
  std::vector<FixedColumnRecord> fixed_columns;
  double objective_shift = 0.0;
  int model_version = 0;
  int last_close_bounds_version = -1;
  int last_activity_checks_version = -1;
  int last_singleton_cols_dual_infer_version = -1;
  int last_parallel_rows_version = -1;
  int last_parallel_cols_version = -1;
  int last_parallel_rows_scan_rows = -1;
  int last_parallel_rows_scan_cols = -1;
  int last_parallel_rows_scan_nnz = -1;
  int last_parallel_cols_scan_rows = -1;
  int last_parallel_cols_scan_cols = -1;
  int last_parallel_cols_scan_nnz = -1;
  int last_mutable_parallel_rows_version = -1;
  int last_mutable_parallel_cols_version = -1;

  MutablePresolveState& equality_state(const PresolveOptions& options) {
    if (!mutable_state) {
      const bool profile = std::getenv("CPU_PRESOLVE_PROFILE") != nullptr;
      std::chrono::steady_clock::time_point start;
      if (profile) {
        std::cerr << "PROFILE_BEGIN\tmutable_state_build"
                  << "\trows=" << current.num_rows()
                  << "\tcols=" << current.num_cols()
                  << "\tnnz=" << current.csc().nnz()
                  << '\n';
        start = std::chrono::steady_clock::now();
      }
      mutable_state = std::make_unique<MutablePresolveState>(
          current, options.zero_tolerance, options.bound_tolerance);
      if (profile) {
        const auto stop = std::chrono::steady_clock::now();
        std::cerr << "PROFILE_END\tmutable_state_build"
                  << "\tstatus=0"
                  << "\tchanged=0"
                  << "\tseconds=" << std::chrono::duration<double>(stop - start).count()
                  << "\tresult_rows=" << current.num_rows()
                  << "\tresult_cols=" << current.num_cols()
                  << "\tresult_nnz=" << current.csc().nnz()
                  << '\n';
      }
    }
    return *mutable_state;
  }

  void invalidate_mutable_state() {
    mutable_state.reset();
  }

  void mark_model_changed() {
    ++model_version;
  }
};

struct RuleOutcome {
  PresolveStatus status = PresolveStatus::kOk;
  bool changed = false;
};

using RuleFunction = PresolveResult (*)(const LpModel&, const PresolveOptions&);
using MutableRuleFunction = MutableReductionOutcome (*)(MutablePresolveState&,
                                                        const PresolveOptions&);

bool profile_rules_enabled() {
  return std::getenv("CPU_PRESOLVE_PROFILE") != nullptr;
}

const char* rule_function_name(RuleFunction rule) {
  if (rule == apply_close_bounds) {
    return "close_bounds";
  }
  if (rule == apply_empty_rows) {
    return "empty_rows";
  }
  if (rule == apply_singleton_rows) {
    return "singleton_rows";
  }
  if (rule == apply_activity_checks) {
    return "activity_checks";
  }
  if (rule == apply_primal_propagation) {
    return "primal_propagation";
  }
  if (rule == apply_parallel_rows) {
    return "parallel_rows";
  }
  if (rule == apply_singleton_cols_dual_infer) {
    return "singleton_cols_dual_infer";
  }
  if (rule == apply_singleton_cols_eq) {
    return "singleton_cols_eq";
  }
  if (rule == apply_doubleton_eq) {
    return "doubleton_eq";
  }
  if (rule == apply_structural_l1_graph_substitution) {
    return "structural_l1_graph_substitution";
  }
  if (rule == apply_structural_l1_substitution) {
    return "structural_l1_substitution";
  }
  if (rule == apply_dual_fix) {
    return "dual_fix";
  }
  if (rule == apply_empty_cols) {
    return "empty_cols";
  }
  if (rule == apply_redundant_bounds) {
    return "redundant_bounds";
  }
  if (rule == apply_parallel_cols) {
    return "parallel_cols";
  }
  return "unknown_rule";
}

double elapsed_seconds(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

PresolveResult finish_scheduler(SchedulerState&& state, PresolveStatus status) {
  return PresolveResult(
      std::move(state.current),
      status,
      state.changed,
      std::move(state.fixed_columns),
      state.objective_shift);
}

RuleOutcome absorb_rule_result(SchedulerState& state, PresolveResult&& result) {
  if (result.status() != PresolveStatus::kOk) {
    return RuleOutcome{result.status(), false};
  }

  state.changed = state.changed || result.changed();
  state.fixed_columns.reserve(state.fixed_columns.size() + result.fixed_columns().size());
  state.fixed_columns.insert(
      state.fixed_columns.end(),
      std::make_move_iterator(result.fixed_columns().begin()),
      std::make_move_iterator(result.fixed_columns().end()));
  state.objective_shift += result.objective_shift();
  if (result.changed()) {
    state.current = std::move(result).reduced_model();
    state.invalidate_mutable_state();
    state.mark_model_changed();
  }
  return RuleOutcome{PresolveStatus::kOk, result.changed()};
}

RuleOutcome run_rule_if_enabled(SchedulerState& state,
                                const PresolveOptions& options,
                                bool enabled,
                                RuleFunction rule) {
  if (!enabled) {
    return RuleOutcome{};
  }
  if (!profile_rules_enabled()) {
    return absorb_rule_result(state, rule(state.current, options));
  }

  const char* name = rule_function_name(rule);
  std::cerr << "PROFILE_BEGIN\t" << name
            << "\trows=" << state.current.num_rows()
            << "\tcols=" << state.current.num_cols()
            << "\tnnz=" << state.current.csc().nnz()
            << '\n';
  const auto start = std::chrono::steady_clock::now();
  PresolveResult result = rule(state.current, options);
  const auto stop = std::chrono::steady_clock::now();
  std::cerr << "PROFILE_END\t" << name
            << "\tstatus=" << static_cast<int>(result.status())
            << "\tchanged=" << (result.changed() ? 1 : 0)
            << "\tseconds=" << elapsed_seconds(start, stop);
  if (result.status() == PresolveStatus::kOk) {
    std::cerr << "\tresult_rows=" << result.reduced_model().num_rows()
              << "\tresult_cols=" << result.reduced_model().num_cols()
              << "\tresult_nnz=" << result.reduced_model().csc().nnz();
  }
  std::cerr << '\n';
  return absorb_rule_result(state, std::move(result));
}

RuleOutcome absorb_mutable_outcome(SchedulerState& state,
                                   MutablePresolveState& mutable_state,
                                   MutableReductionOutcome&& outcome) {
  if (outcome.status != PresolveStatus::kOk) {
    return RuleOutcome{outcome.status, false};
  }
  if (outcome.reductions == 0) {
    return RuleOutcome{};
  }

  const double shift = mutable_state.objective_delta;
  const bool profile = profile_rules_enabled();
  std::chrono::steady_clock::time_point export_start;
  if (profile) {
    std::cerr << "PROFILE_BEGIN\tmutable_export"
              << "\trows=" << mutable_state.reduced_rows
              << "\tcols=" << mutable_state.num_cols -
                     static_cast<int>(std::count(
                         mutable_state.active_col.begin(),
                         mutable_state.active_col.end(),
                         static_cast<unsigned char>(0)))
              << "\tnnz=" << mutable_state.active_nnz()
              << '\n';
    export_start = std::chrono::steady_clock::now();
  }
  state.current = mutable_state.export_model(mutable_state.active_column_map());
  if (profile) {
    const auto export_stop = std::chrono::steady_clock::now();
    std::cerr << "PROFILE_END\tmutable_export"
              << "\tstatus=0"
              << "\tchanged=1"
              << "\tseconds=" << elapsed_seconds(export_start, export_stop)
              << "\tresult_rows=" << state.current.num_rows()
              << "\tresult_cols=" << state.current.num_cols()
              << "\tresult_nnz=" << state.current.csc().nnz()
              << '\n';
  }
  mutable_state.obj_constant += shift;
  mutable_state.objective_delta = 0.0;
  state.changed = true;
  state.objective_shift += shift;
  state.fixed_columns.reserve(state.fixed_columns.size() + outcome.fixed_columns.size());
  state.fixed_columns.insert(
      state.fixed_columns.end(),
      std::make_move_iterator(outcome.fixed_columns.begin()),
      std::make_move_iterator(outcome.fixed_columns.end()));
  state.mark_model_changed();
  return RuleOutcome{PresolveStatus::kOk, true};
}

RuleOutcome run_mutable_rule_without_export(SchedulerState& state,
                                            MutablePresolveState& mutable_state,
                                            const PresolveOptions& options,
                                            bool enabled,
                                            const char* name,
                                            MutableRuleFunction rule,
                                            MutableReductionOutcome& combined) {
  if (!enabled) {
    return RuleOutcome{};
  }

  const bool profile = profile_rules_enabled();
  std::chrono::steady_clock::time_point start;
  if (profile) {
    std::cerr << "PROFILE_BEGIN\t" << name
              << "\trows=" << state.current.num_rows()
              << "\tcols=" << state.current.num_cols()
              << "\tnnz=" << state.current.csc().nnz()
              << '\n';
    start = std::chrono::steady_clock::now();
  }

  MutableReductionOutcome outcome = rule(mutable_state, options);
  const PresolveStatus status = outcome.status;
  const bool changed = status == PresolveStatus::kOk && outcome.reductions > 0;
  if (status == PresolveStatus::kOk) {
    combined.reductions += outcome.reductions;
    combined.fixed_columns.reserve(combined.fixed_columns.size() + outcome.fixed_columns.size());
    combined.fixed_columns.insert(
        combined.fixed_columns.end(),
        std::make_move_iterator(outcome.fixed_columns.begin()),
        std::make_move_iterator(outcome.fixed_columns.end()));
  }

  if (profile) {
    const auto stop = std::chrono::steady_clock::now();
    std::cerr << "PROFILE_END\t" << name
              << "\tstatus=" << static_cast<int>(status)
              << "\tchanged=" << (changed ? 1 : 0)
              << "\tseconds=" << elapsed_seconds(start, stop);
    if (std::strcmp(name, "singleton_cols") == 0) {
      std::cerr << "\tmode=single_queue";
    }
    if (status == PresolveStatus::kOk) {
      std::cerr << "\tresult_rows=" << mutable_state.reduced_rows
                << "\tresult_cols=" << mutable_state.num_cols -
                       static_cast<int>(std::count(
                           mutable_state.active_col.begin(),
                           mutable_state.active_col.end(),
                           static_cast<unsigned char>(0)))
                << "\tresult_nnz=" << mutable_state.active_nnz();
    }
    std::cerr << '\n';
  }

  if (status != PresolveStatus::kOk) {
    return RuleOutcome{status, false};
  }
  return RuleOutcome{PresolveStatus::kOk, changed};
}

bool append_mutable_outcome(MutableReductionOutcome& combined,
                            MutableReductionOutcome&& outcome) {
  if (outcome.status != PresolveStatus::kOk) {
    combined.status = outcome.status;
    return false;
  }
  combined.reductions += outcome.reductions;
  combined.fixed_columns.reserve(combined.fixed_columns.size() + outcome.fixed_columns.size());
  combined.fixed_columns.insert(
      combined.fixed_columns.end(),
      std::make_move_iterator(outcome.fixed_columns.begin()),
      std::make_move_iterator(outcome.fixed_columns.end()));
  return true;
}

MutableReductionOutcome process_trivial_cleanup_mutable(MutablePresolveState& state,
                                                        const PresolveOptions& options) {
  MutableReductionOutcome combined;

  auto run = [&](bool enabled, MutableRuleFunction rule, bool& round_changed) {
    if (!enabled) {
      return true;
    }
    MutableReductionOutcome outcome = rule(state, options);
    const bool rule_changed = outcome.status == PresolveStatus::kOk && outcome.reductions > 0;
    if (!append_mutable_outcome(combined, std::move(outcome))) {
      return false;
    }
    round_changed = round_changed || rule_changed;
    return true;
  };

  bool round_changed = false;
  do {
    round_changed = false;
    if (!run(options.enable_close_bounds, process_close_bounds_mutable, round_changed)) {
      return combined;
    }
    if (!run(options.enable_empty_cols, process_empty_cols_mutable, round_changed)) {
      return combined;
    }
    if (!run(options.enable_dual_fix, process_dual_fix_mutable, round_changed)) {
      return combined;
    }
    if (!run(options.enable_singleton_rows, process_singleton_rows_mutable, round_changed)) {
      return combined;
    }
    if (!run(options.enable_empty_rows, process_empty_rows_mutable, round_changed)) {
      return combined;
    }
    if (!run(options.enable_empty_cols, process_empty_cols_mutable, round_changed)) {
      return combined;
    }
  } while (round_changed);

  return combined;
}

RuleOutcome run_mutable_cleanup_without_export(SchedulerState& state,
                                               MutablePresolveState& mutable_state,
                                               const PresolveOptions& options,
                                               MutableReductionOutcome& combined) {
  const bool enabled =
      options.enable_close_bounds ||
      options.enable_empty_cols ||
      options.enable_dual_fix ||
      options.enable_singleton_rows ||
      options.enable_empty_rows;
  return run_mutable_rule_without_export(
      state,
      mutable_state,
      options,
      enabled,
      "trivial_cleanup",
      process_trivial_cleanup_mutable,
      combined);
}

bool add_changed_or_stop(RuleOutcome outcome, bool& changed, PresolveStatus& status) {
  if (outcome.status != PresolveStatus::kOk) {
    status = outcome.status;
    return false;
  }
  changed = changed || outcome.changed;
  return true;
}

RuleOutcome run_cleanup_tier(SchedulerState& state, const PresolveOptions& options) {
  bool changed = false;
  PresolveStatus status = PresolveStatus::kOk;

  MutablePresolveState& mutable_state = state.equality_state(options);
  MutableReductionOutcome combined;
  if (!add_changed_or_stop(
          run_mutable_cleanup_without_export(state, mutable_state, options, combined),
          changed,
          status)) {
    return RuleOutcome{status, changed};
  }

  RuleOutcome export_outcome =
      absorb_mutable_outcome(state, mutable_state, std::move(combined));
  if (!add_changed_or_stop(export_outcome, changed, status)) {
    return RuleOutcome{status, changed};
  }
  return RuleOutcome{PresolveStatus::kOk, changed};
}

RuleOutcome run_structural_l1_tier(SchedulerState& state, const PresolveOptions& options) {
  RuleOutcome structural_outcome = run_rule_if_enabled(
      state,
      options,
      options.enable_structural_l1_substitution,
      apply_structural_l1_substitution);
  if (structural_outcome.status != PresolveStatus::kOk || !structural_outcome.changed) {
    return structural_outcome;
  }

  RuleOutcome cleanup_outcome = run_cleanup_tier(state, options);
  if (cleanup_outcome.status != PresolveStatus::kOk) {
    return RuleOutcome{cleanup_outcome.status, true};
  }
  return RuleOutcome{PresolveStatus::kOk, true};
}

RuleOutcome run_mutable_core_scheduler(SchedulerState& state, const PresolveOptions& options) {
  MutablePresolveState& mutable_state = state.equality_state(options);
  MutableReductionOutcome combined;
  bool changed = false;
  PresolveStatus status = PresolveStatus::kOk;
  constexpr double kMeaningfulNnzReductionRatio = 0.95;

  auto add = [&](RuleOutcome outcome, bool& local_changed) {
    if (outcome.status != PresolveStatus::kOk) {
      status = outcome.status;
      return false;
    }
    local_changed = local_changed || outcome.changed;
    changed = changed || outcome.changed;
    return true;
  };

  auto run_fast_phase = [&]() {
    bool rule_changed = false;
    const bool singleton_enabled =
        options.enable_singleton_cols_dual_infer || options.enable_singleton_cols_eq;
    if (!add(
            run_mutable_rule_without_export(
                state,
                mutable_state,
                options,
                singleton_enabled,
                "singleton_cols",
                process_singleton_cols_combined_mutable,
                combined),
            rule_changed)) {
      return false;
    }
    if (singleton_enabled) {
      bool cleanup_changed = false;
      if (!add(
              run_mutable_cleanup_without_export(state, mutable_state, options, combined),
              cleanup_changed)) {
        return false;
      }
    }

    rule_changed = false;
    if (options.enable_doubleton_eq) {
      if (!add(
              run_mutable_rule_without_export(
                  state,
                  mutable_state,
                  options,
                  true,
                  "doubleton_eq",
                  process_doubleton_eq,
                  combined),
              rule_changed)) {
        return false;
      }
      bool cleanup_changed = false;
      if (!add(
              run_mutable_cleanup_without_export(state, mutable_state, options, combined),
              cleanup_changed)) {
        return false;
      }
    }
    return true;
  };

  auto run_medium_phase = [&]() {
    const bool propagation_enabled =
        options.enable_activity_checks || options.enable_primal_propagation;
    if (propagation_enabled) {
      bool rule_changed = false;
      if (!add(
              run_mutable_rule_without_export(
                  state,
                  mutable_state,
                  options,
                  options.enable_activity_checks,
                  "activity_checks",
                  process_activity_checks_mutable,
                  combined),
              rule_changed)) {
        return false;
      }

      rule_changed = false;
      if (!add(
              run_mutable_rule_without_export(
                  state,
                  mutable_state,
                  options,
                  options.enable_primal_propagation,
                  "primal_propagation",
                  process_primal_propagation_dirty,
                  combined),
              rule_changed)) {
        return false;
      }

      bool cleanup_changed = false;
      if (!add(
              run_mutable_cleanup_without_export(state, mutable_state, options, combined),
              cleanup_changed)) {
        return false;
      }
    }

    bool rule_changed = false;
    const bool should_run_parallel_rows =
        options.enable_parallel_rows &&
        state.last_mutable_parallel_rows_version != mutable_state.structural_change_version;
    if (should_run_parallel_rows) {
      if (!add(
              run_mutable_rule_without_export(
                  state,
                  mutable_state,
                  options,
                  true,
                  "parallel_rows",
                  process_parallel_rows_mutable,
                  combined),
              rule_changed)) {
        return false;
      }
      state.last_mutable_parallel_rows_version = mutable_state.structural_change_version;
    }

    rule_changed = false;
    const bool should_run_parallel_cols =
        options.enable_parallel_cols &&
        state.last_mutable_parallel_cols_version != mutable_state.structural_change_version;
    if (should_run_parallel_cols) {
      if (!add(
              run_mutable_rule_without_export(
                  state,
                  mutable_state,
                  options,
                  true,
                  "parallel_cols",
                  process_parallel_cols_mutable,
                  combined),
              rule_changed)) {
        return false;
      }
      state.last_mutable_parallel_cols_version = mutable_state.structural_change_version;
      bool cleanup_changed = false;
      if (!add(
              run_mutable_cleanup_without_export(state, mutable_state, options, combined),
              cleanup_changed)) {
        return false;
      }
    }
    return true;
  };

  enum class ComplexityPhase {
    kFast,
    kMedium,
  };

  const int max_phases = std::max(1, options.max_iterations) * 2;
  ComplexityPhase phase = ComplexityPhase::kFast;
  int nnz_before_cycle = mutable_state.active_nnz();
  for (int phase_count = 0; phase_count < max_phases; ++phase_count) {
    bool cleanup_changed = false;
    if (!add(
            run_mutable_cleanup_without_export(state, mutable_state, options, combined),
            cleanup_changed)) {
      return RuleOutcome{status, changed};
    }

    const int nnz_before_phase = mutable_state.active_nnz();
    if (phase == ComplexityPhase::kFast) {
      if (!run_fast_phase()) {
        return RuleOutcome{status, changed};
      }
      const int nnz_after_phase = mutable_state.active_nnz();
      if (static_cast<double>(nnz_after_phase) <
          kMeaningfulNnzReductionRatio * static_cast<double>(nnz_before_phase)) {
        phase = ComplexityPhase::kFast;
      } else {
        phase = ComplexityPhase::kMedium;
      }
      continue;
    }

    if (!run_medium_phase()) {
      return RuleOutcome{status, changed};
    }
    const int nnz_after_cycle = mutable_state.active_nnz();
    if (nnz_after_cycle == nnz_before_phase) {
      break;
    }
    if (static_cast<double>(nnz_after_cycle) >=
        kMeaningfulNnzReductionRatio * static_cast<double>(nnz_before_cycle)) {
      break;
    }
    nnz_before_cycle = nnz_after_cycle;
    phase = ComplexityPhase::kFast;
  }

  RuleOutcome export_outcome =
      absorb_mutable_outcome(state, mutable_state, std::move(combined));
  if (export_outcome.status != PresolveStatus::kOk) {
    return export_outcome;
  }
  changed = changed || export_outcome.changed;
  return RuleOutcome{PresolveStatus::kOk, changed};
}

PresolveResult run_fixed_scheduler(const LpModel& model, const PresolveOptions& options) {
  SchedulerState state(model);
  PresolveStatus status = PresolveStatus::kOk;
  bool ignored_changed = false;

  if (options.enable_close_bounds) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_close_bounds);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_structural_l1_substitution) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_structural_l1_substitution);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_empty_rows) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_empty_rows);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_singleton_rows) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_singleton_rows);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_activity_checks) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_activity_checks);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_primal_propagation) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_primal_propagation);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_parallel_rows) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_parallel_rows);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_empty_cols) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_empty_cols);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_singleton_cols_dual_infer) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_singleton_cols_dual_infer);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_singleton_cols_eq) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_singleton_cols_eq);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_doubleton_eq) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_doubleton_eq);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_dual_fix) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_dual_fix);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_redundant_bounds) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_redundant_bounds);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  if (options.enable_parallel_cols) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_parallel_cols);
    if (!add_changed_or_stop(outcome, ignored_changed, status)) {
      return finish_scheduler(std::move(state), status);
    }
  }

  return finish_scheduler(std::move(state), PresolveStatus::kOk);
}

RuleOutcome run_tiered_bootstrap_rules(SchedulerState& state, const PresolveOptions& options) {
  PresolveStatus status = PresolveStatus::kOk;
  bool changed = false;
  auto run = [&](bool enabled, RuleFunction rule) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, enabled, rule);
    return add_changed_or_stop(outcome, changed, status);
  };

  if (!run(options.enable_empty_rows, apply_empty_rows)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_singleton_rows, apply_singleton_rows)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_activity_checks, apply_activity_checks)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_parallel_rows, apply_parallel_rows)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_close_bounds, apply_close_bounds)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_empty_cols, apply_empty_cols)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_singleton_cols_dual_infer, apply_singleton_cols_dual_infer)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_singleton_cols_eq, apply_singleton_cols_eq)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_dual_fix, apply_dual_fix)) {
    return RuleOutcome{status, changed};
  }
  if (!run(options.enable_parallel_cols, apply_parallel_cols)) {
    return RuleOutcome{status, changed};
  }

  return RuleOutcome{PresolveStatus::kOk, changed};
}

PresolveResult run_tiered_scheduler(const LpModel& model, const PresolveOptions& options) {
  SchedulerState state(model);

  RuleOutcome structural_outcome = run_structural_l1_tier(state, options);
  if (structural_outcome.status != PresolveStatus::kOk) {
    return finish_scheduler(std::move(state), structural_outcome.status);
  }

  RuleOutcome bootstrap_outcome = run_tiered_bootstrap_rules(state, options);
  if (bootstrap_outcome.status != PresolveStatus::kOk) {
    return finish_scheduler(std::move(state), bootstrap_outcome.status);
  }
  if (bootstrap_outcome.changed) {
    RuleOutcome structural_after_bootstrap = run_structural_l1_tier(state, options);
    if (structural_after_bootstrap.status != PresolveStatus::kOk) {
      return finish_scheduler(std::move(state), structural_after_bootstrap.status);
    }
  }

  RuleOutcome core_outcome = run_mutable_core_scheduler(state, options);
  if (core_outcome.status != PresolveStatus::kOk) {
    return finish_scheduler(std::move(state), core_outcome.status);
  }

  if (options.enable_redundant_bounds) {
    RuleOutcome outcome = run_rule_if_enabled(state, options, true, apply_redundant_bounds);
    if (outcome.status != PresolveStatus::kOk) {
      return finish_scheduler(std::move(state), outcome.status);
    }
  }

  return finish_scheduler(std::move(state), PresolveStatus::kOk);
}

}  // namespace

PresolveResult Presolver::run(const LpModel& model, const PresolveOptions& options) {
  if (options.scheduler == PresolveScheduler::kFixed) {
    return run_fixed_scheduler(model, options);
  }
  return run_tiered_scheduler(model, options);
}

}  // namespace cpu_presolve
