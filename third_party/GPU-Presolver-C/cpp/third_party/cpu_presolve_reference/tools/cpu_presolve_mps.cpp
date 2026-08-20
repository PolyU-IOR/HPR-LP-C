#include "cpu_presolve/mpsreader.hpp"
#include "cpu_presolve/presolver.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

const char* status_name(cpu_presolve::PresolveStatus status) {
  switch (status) {
    case cpu_presolve::PresolveStatus::kOk:
      return "ok";
    case cpu_presolve::PresolveStatus::kInfeasible:
      return "infeasible";
    case cpu_presolve::PresolveStatus::kUnbounded:
      return "unbounded";
  }
  return "unknown";
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

bool env_bool(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  const std::string text(value);
  if (text == "1" || text == "true" || text == "TRUE" || text == "yes" || text == "YES" ||
      text == "on" || text == "ON") {
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE" || text == "no" || text == "NO" ||
      text == "off" || text == "OFF") {
    return false;
  }
  return default_value;
}

int env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    return default_value;
  }
}

void apply_env_options(cpu_presolve::PresolveOptions& options) {
  if (const char* scheduler = std::getenv("GPU_PRESOLVER_SCHEDULER")) {
    const std::string value(scheduler);
    if (value == "fixed") {
      options.scheduler = cpu_presolve::PresolveScheduler::kFixed;
    } else if (value == "tiered") {
      options.scheduler = cpu_presolve::PresolveScheduler::kTiered;
    }
  }

  options.enable_close_bounds =
      env_bool("GPU_PRESOLVER_ENABLE_CLOSE_BOUNDS", options.enable_close_bounds);
  options.enable_empty_rows =
      env_bool("GPU_PRESOLVER_ENABLE_EMPTY_ROWS", options.enable_empty_rows);
  options.enable_singleton_rows =
      env_bool("GPU_PRESOLVER_ENABLE_SINGLETON_ROWS", options.enable_singleton_rows);
  options.enable_activity_checks =
      env_bool("GPU_PRESOLVER_ENABLE_ACTIVITY_CHECKS", options.enable_activity_checks);
  options.enable_primal_propagation =
      env_bool("GPU_PRESOLVER_ENABLE_PRIMAL_PROPAGATION", options.enable_primal_propagation);
  options.enable_parallel_rows =
      env_bool("GPU_PRESOLVER_ENABLE_PARALLEL_ROWS", options.enable_parallel_rows);
  options.enable_dual_fix =
      env_bool("GPU_PRESOLVER_ENABLE_DUAL_FIX", options.enable_dual_fix);
  options.enable_empty_cols =
      env_bool("GPU_PRESOLVER_ENABLE_EMPTY_COLS", options.enable_empty_cols);
  options.enable_singleton_cols_dual_infer = env_bool(
      "GPU_PRESOLVER_ENABLE_SINGLETON_COLS_DUAL_INFER",
      options.enable_singleton_cols_dual_infer);
  options.enable_singleton_cols_eq =
      env_bool("GPU_PRESOLVER_ENABLE_SINGLETON_COLS_EQ", options.enable_singleton_cols_eq);
  options.enable_doubleton_eq =
      env_bool("GPU_PRESOLVER_ENABLE_DOUBLETON_EQ", options.enable_doubleton_eq);
  options.enable_redundant_bounds =
      env_bool("GPU_PRESOLVER_ENABLE_REDUNDANT_BOUNDS", options.enable_redundant_bounds);
  options.enable_parallel_cols =
      env_bool("GPU_PRESOLVER_ENABLE_PARALLEL_COLS", options.enable_parallel_cols);
  options.enable_structural_l1_substitution = env_bool(
      "GPU_PRESOLVER_ENABLE_STRUCTURAL_L1_SUBSTITUTION",
      options.enable_structural_l1_substitution);
  options.doubleton_eq_max_reductions = env_int(
      "GPU_PRESOLVER_DOUBLETON_MAX_REDUCTIONS",
      options.doubleton_eq_max_reductions);
  options.doubleton_eq_max_fill_in_proxy = env_int(
      "GPU_PRESOLVER_DOUBLETON_MAX_FILL_IN_PROXY",
      options.doubleton_eq_max_fill_in_proxy);
  options.doubleton_eq_min_selected_per_batch = env_int(
      "GPU_PRESOLVER_DOUBLETON_MIN_SELECTED_PER_BATCH",
      options.doubleton_eq_min_selected_per_batch);
  options.doubleton_eq_batch_mode =
      env_bool("GPU_PRESOLVER_DOUBLETON_BATCH_MODE", true);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: cpu_presolve_mps <model.mps|->\n";
    return 2;
  }

  try {
    std::unique_ptr<std::ifstream> file;
    std::istream* input = &std::cin;
    const std::string path = argv[1];
    if (path != "-") {
      file = std::make_unique<std::ifstream>(path);
      if (!file->is_open()) {
        std::cerr << "failed to open " << path << '\n';
        return 2;
      }
      input = file.get();
    }

    const auto parse_start = std::chrono::steady_clock::now();
    cpu_presolve::MpsModel model = cpu_presolve::read_mps(*input);
    const auto parse_stop = std::chrono::steady_clock::now();

    cpu_presolve::PresolveOptions options;
    apply_env_options(options);
    const auto presolve_start = std::chrono::steady_clock::now();
    cpu_presolve::PresolveResult result = cpu_presolve::Presolver::run(model.lp, options);
    const auto presolve_stop = std::chrono::steady_clock::now();

    const cpu_presolve::LpModel& reduced = result.reduced_model();
    std::cout << "model " << model.name << '\n';
    std::cout << "status " << status_name(result.status()) << '\n';
    std::cout << "changed " << (result.changed() ? 1 : 0) << '\n';
    std::cout << "original_rows " << model.lp.num_rows() << '\n';
    std::cout << "original_cols " << model.lp.num_cols() << '\n';
    std::cout << "original_nnz " << model.lp.csc().nnz() << '\n';
    std::cout << "reduced_rows " << reduced.num_rows() << '\n';
    std::cout << "reduced_cols " << reduced.num_cols() << '\n';
    std::cout << "reduced_nnz " << reduced.csc().nnz() << '\n';
    std::cout << "fixed_columns " << result.fixed_columns().size() << '\n';
    std::cout << "objective_shift " << result.objective_shift() << '\n';
    std::cout << "parse_seconds " << seconds_since(parse_start, parse_stop) << '\n';
    std::cout << "presolve_seconds " << seconds_since(presolve_start, presolve_stop) << '\n';
    return result.status() == cpu_presolve::PresolveStatus::kOk ? 0 : 1;
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << '\n';
    return 1;
  }
}
