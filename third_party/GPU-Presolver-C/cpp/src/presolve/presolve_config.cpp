#include "gpu_presolver/presolve/presolve_config.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace gpu_presolver::presolve {
namespace {

std::string trim(const std::string& text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::string strip_comment(const std::string& line) {
  bool quoted = false;
  bool escaped = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (quoted && ch == '\\' && !escaped) {
      escaped = true;
      continue;
    }
    if (ch == '"' && !escaped) {
      quoted = !quoted;
    } else if (ch == '#' && !quoted) {
      return line.substr(0, i);
    }
    escaped = false;
  }
  return line;
}

[[noreturn]] void config_error(const std::string& path,
                               int line_number,
                               const std::string& message) {
  throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + message);
}

bool parse_bool(const std::string& value,
                const std::string& path,
                int line_number,
                const std::string& key) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  config_error(path, line_number, key + " requires true or false");
}

int parse_int(const std::string& value,
              const std::string& path,
              int line_number,
              const std::string& key) {
  int result = 0;
  const char* first = value.data();
  const char* last = first + value.size();
  const auto parsed = std::from_chars(first, last, result);
  if (parsed.ec != std::errc{} || parsed.ptr != last) {
    config_error(path, line_number, key + " requires an integer");
  }
  return result;
}

int parse_nonnegative_int(const std::string& value,
                          const std::string& path,
                          int line_number,
                          const std::string& key) {
  const int result = parse_int(value, path, line_number, key);
  if (result < 0) {
    config_error(path, line_number, key + " requires a nonnegative integer");
  }
  return result;
}

double parse_double(const std::string& value,
                    const std::string& path,
                    int line_number,
                    const std::string& key,
                    bool allow_infinity = false) {
  double result = 0.0;
  std::size_t consumed = 0;
  try {
    result = std::stod(value, &consumed);
  } catch (const std::exception&) {
    config_error(path, line_number, key + " requires a number");
  }
  if (consumed != value.size() || std::isnan(result) || (!allow_infinity && !std::isfinite(result))) {
    config_error(path, line_number, key + " requires a valid number");
  }
  return result;
}

double parse_nonnegative_double(const std::string& value,
                                const std::string& path,
                                int line_number,
                                const std::string& key,
                                bool allow_infinity = false) {
  const double result = parse_double(value, path, line_number, key, allow_infinity);
  if (result < 0.0) {
    config_error(path, line_number, key + " requires a nonnegative number");
  }
  return result;
}

double parse_positive_double(const std::string& value,
                             const std::string& path,
                             int line_number,
                             const std::string& key) {
  const double result = parse_double(value, path, line_number, key);
  if (result <= 0.0) {
    config_error(path, line_number, key + " requires a positive number");
  }
  return result;
}

std::string parse_string(const std::string& value,
                         const std::string& path,
                         int line_number,
                         const std::string& key) {
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    config_error(path, line_number, key + " requires a quoted string");
  }
  return value.substr(1, value.size() - 2);
}

void apply_value(PresolveParams& params,
                 const std::string& key,
                 const std::string& value,
                 const std::string& path,
                 int line_number) {
  if (key == "runtime.verbose") {
    params.verbose = parse_bool(value, path, line_number, key);
  } else if (key == "runtime.debug_checks") {
    params.debug_checks = parse_bool(value, path, line_number, key);
  } else if (key == "runtime.trace_enabled") {
    params.trace_enabled = parse_bool(value, path, line_number, key);
  } else if (key == "runtime.record_postsolve_tape") {
    params.record_postsolve_tape = parse_bool(value, path, line_number, key);
  } else if (key == "runtime.record_postsolve_tape_cpu") {
    params.record_postsolve_tape_cpu = parse_bool(value, path, line_number, key);
  } else if (key == "limits.max_presolve_iters") {
    params.max_iters = parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "limits.max_presolve_time") {
    params.max_time = parse_nonnegative_double(value, path, line_number, key, true);
  } else if (key == "tolerances.feasibility") {
    params.feasibility_tol = parse_positive_double(value, path, line_number, key);
  } else if (key == "tolerances.bound") {
    params.bound_tol = parse_positive_double(value, path, line_number, key);
  } else if (key == "tolerances.zero") {
    params.zero_tol = parse_positive_double(value, path, line_number, key);
  } else if (key == "tolerances.primal_propagation_min_tighten_abs") {
    params.primal_propagation_min_tighten_abs =
        parse_nonnegative_double(value, path, line_number, key);
  } else if (key == "folding.enabled") {
    params.enable_folding = parse_bool(value, path, line_number, key);
  } else if (key == "folding.tolerance") {
    params.folding_tolerance = parse_positive_double(value, path, line_number, key);
  } else if (key == "rules.close_bounds") {
    params.enable_close_bounds = parse_bool(value, path, line_number, key);
  } else if (key == "rules.empty_rows") {
    params.enable_empty_rows = parse_bool(value, path, line_number, key);
  } else if (key == "rules.singleton_rows") {
    params.enable_singleton_rows = parse_bool(value, path, line_number, key);
  } else if (key == "rules.activity_checks") {
    params.enable_activity_checks = parse_bool(value, path, line_number, key);
  } else if (key == "rules.primal_propagation") {
    params.enable_primal_propagation = parse_bool(value, path, line_number, key);
  } else if (key == "rules.parallel_rows") {
    params.enable_parallel_rows = parse_bool(value, path, line_number, key);
  } else if (key == "rules.empty_cols") {
    params.enable_empty_cols = parse_bool(value, path, line_number, key);
  } else if (key == "rules.singleton_cols_eq") {
    params.enable_singleton_cols_eq = parse_bool(value, path, line_number, key);
  } else if (key == "rules.singleton_cols_dual_infer") {
    params.enable_singleton_cols_dual_infer = parse_bool(value, path, line_number, key);
  } else if (key == "rules.doubleton_eq") {
    params.enable_doubleton_eq = parse_bool(value, path, line_number, key);
  } else if (key == "rules.linear_eq_agg") {
    params.enable_linear_eq_agg = parse_bool(value, path, line_number, key);
  } else if (key == "rules.dual_fix") {
    params.enable_dual_fix = parse_bool(value, path, line_number, key);
  } else if (key == "rules.parallel_cols") {
    params.enable_parallel_cols = parse_bool(value, path, line_number, key);
  } else if (key == "rules.redundant_bounds") {
    params.enable_redundant_bounds = parse_bool(value, path, line_number, key);
  } else if (key == "rules.fme_projection") {
    params.enable_fme_projection = parse_bool(value, path, line_number, key);
  } else if (key == "rules.structural_l1_substitution") {
    params.enable_structural_l1_substitution = parse_bool(value, path, line_number, key);
  } else if (key == "structural_l1.residual_bound_as_free_min") {
    params.structural_l1_residual_bound_as_free_min =
        parse_positive_double(value, path, line_number, key);
  } else if (key == "doubleton.single_batch_per_iter") {
    params.doubleton_eq_single_batch_per_iter = parse_bool(value, path, line_number, key);
  } else if (key == "doubleton.max_fill_in_proxy") {
    params.doubleton_eq_max_fill_in_proxy =
        parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "doubleton.scan") {
    params.doubleton_eq_scan = parse_bool(value, path, line_number, key);
  } else if (key == "doubleton.min_selected_per_batch") {
    params.doubleton_eq_min_selected_per_batch =
        parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "doubleton.min_selected_ratio") {
    params.doubleton_eq_min_selected_ratio =
        parse_nonnegative_double(value, path, line_number, key);
  } else if (key == "doubleton.max_batch_rounds") {
    params.doubleton_eq_max_batch_rounds =
        parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "doubleton.max_time") {
    params.doubleton_eq_max_time = parse_nonnegative_double(value, path, line_number, key);
  } else if (key == "scheduler.mode") {
    const std::string mode = parse_string(value, path, line_number, key);
    if (mode == "tiered") {
      params.use_tiered_scheduler = true;
    } else if (mode == "fixed") {
      params.use_tiered_scheduler = false;
    } else {
      config_error(path, line_number, key + " requires \"tiered\" or \"fixed\"");
    }
  } else if (key == "scheduler.tiered_bootstrap") {
    params.enable_tiered_bootstrap = parse_bool(value, path, line_number, key);
  } else if (key == "scheduler.cleanup_max_rounds") {
    params.tiered_cleanup_max_rounds = parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "scheduler.light_continue_ratio") {
    params.tiered_light_continue_ratio = parse_nonnegative_double(value, path, line_number, key);
  } else if (key == "scheduler.cycle_stop_ratio") {
    params.tiered_cycle_stop_ratio = parse_nonnegative_double(value, path, line_number, key);
  } else if (key == "scheduler.max_light_streak") {
    params.tiered_max_light_streak = parse_nonnegative_int(value, path, line_number, key);
  } else if (key == "scheduler.global_period") {
    params.tiered_global_period = parse_nonnegative_int(value, path, line_number, key);
  } else {
    config_error(path, line_number, "unknown presolve setting: " + key);
  }
}

}  // namespace

void load_presolve_params_from_toml(const std::string& path, PresolveParams& params) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open presolve config: " + path);
  }

  PresolveParams loaded = params;
  std::string section;
  std::string raw_line;
  int line_number = 0;
  while (std::getline(input, raw_line)) {
    ++line_number;
    const std::string line = trim(strip_comment(raw_line));
    if (line.empty()) {
      continue;
    }
    if (line.front() == '[') {
      if (line.size() < 3 || line.back() != ']') {
        config_error(path, line_number, "invalid TOML section");
      }
      section = trim(line.substr(1, line.size() - 2));
      if (section.empty()) {
        config_error(path, line_number, "empty TOML section");
      }
      continue;
    }

    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) {
      config_error(path, line_number, "expected key = value");
    }
    if (section.empty()) {
      config_error(path, line_number, "presolve settings must be inside a section");
    }
    const std::string name = trim(line.substr(0, equals));
    const std::string value = trim(line.substr(equals + 1));
    if (name.empty() || value.empty()) {
      config_error(path, line_number, "expected key = value");
    }
    apply_value(loaded, section + "." + name, value, path, line_number);
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading presolve config: " + path);
  }
  params = loaded;
}

}  // namespace gpu_presolver::presolve
