#include "gpu_presolver/presolve/presolve_config.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream output(path);
  assert(output.is_open());
  output << contents;
  assert(output.good());
}

}  // namespace

int main() {
  namespace presolve = gpu_presolver::presolve;

#ifdef GPU_PRESOLVER_SOURCE_DIR
  presolve::PresolveParams default_config;
  presolve::load_presolve_params_from_toml(
      std::string(GPU_PRESOLVER_SOURCE_DIR) + "/../config/default.toml", default_config);
  assert(default_config.enable_folding);
  assert(default_config.folding_tolerance == 1e-8);
  assert(default_config.max_iters == 10);
  assert(default_config.use_tiered_scheduler);
#endif

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "gpu_presolver_config_test";
  std::filesystem::create_directories(directory);

  const std::filesystem::path valid_path = directory / "valid.toml";
  write_file(valid_path,
             "[runtime]\n"
             "verbose = true # inline comment\n"
             "record_postsolve_tape_cpu = true\n"
             "[limits]\n"
             "max_presolve_iters = 17\n"
             "max_presolve_time = inf\n"
             "[tolerances]\n"
             "feasibility = 2e-7\n"
             "primal_propagation_min_tighten_abs = 0.25\n"
             "[folding]\n"
             "enabled = false\n"
             "tolerance = 3e-9\n"
             "[rules]\n"
             "doubleton_eq = false\n"
             "redundant_bounds = true\n"
             "[doubleton]\n"
             "single_batch_per_iter = true\n"
             "max_fill_in_proxy = 42\n"
             "[scheduler]\n"
             "mode = \"fixed\"\n"
             "global_period = 7\n");

  presolve::PresolveParams params;
  params.enable_folding = true;
  params.bound_tol = 9e-6;
  presolve::load_presolve_params_from_toml(valid_path.string(), params);
  assert(params.verbose);
  assert(params.record_postsolve_tape_cpu);
  assert(params.max_iters == 17);
  assert(params.max_time > 1e100);
  assert(params.feasibility_tol == 2e-7);
  assert(params.bound_tol == 9e-6);  // Missing keys retain their existing value.
  assert(params.primal_propagation_min_tighten_abs == 0.25);
  assert(!params.enable_folding);
  assert(params.folding_tolerance == 3e-9);
  assert(!params.enable_doubleton_eq);
  assert(params.enable_redundant_bounds);
  assert(params.doubleton_eq_single_batch_per_iter);
  assert(params.doubleton_eq_max_fill_in_proxy == 42);
  assert(!params.use_tiered_scheduler);
  assert(params.tiered_global_period == 7);

  const std::filesystem::path invalid_path = directory / "invalid.toml";
  write_file(invalid_path, "[rules]\nmisspelled_rule = true\n");
  const presolve::PresolveParams before = params;
  bool threw = false;
  try {
    presolve::load_presolve_params_from_toml(invalid_path.string(), params);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(params.max_iters == before.max_iters);
  assert(params.enable_folding == before.enable_folding);

  std::filesystem::remove_all(directory);
  return 0;
}
