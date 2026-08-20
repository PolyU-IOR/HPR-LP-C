#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdlib>

namespace {

void apply_env_overrides(gpu_presolver::presolve::PresolveParams& params) {
  gpu_presolver::tools::apply_presolve_env_overrides(params);
}

void write_lp_export(const std::filesystem::path& out_dir,
                     const std::string& model_name,
                     const gpu_presolver::presolve::GpuPresolveSummary& summary,
                     std::int32_t original_rows,
                     std::int32_t original_cols,
                     bool presolve_enabled,
                     bool folding_applied,
                     std::int32_t folded_rows,
                     std::int32_t folded_cols) {
  namespace tools = gpu_presolver::tools;
  const gpu_presolver::presolve::LPInfoGpu& lp = summary.reduced_lp;
  std::filesystem::create_directories(out_dir);

  std::ofstream meta(out_dir / "meta.txt");
  if (!meta.is_open()) {
    throw std::runtime_error("failed to open export meta");
  }
  meta << "format gpu_presolver_reduced_lp_v1\n";
  meta << "model " << model_name << "\n";
  meta << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
  meta << "original_rows " << original_rows << "\n";
  meta << "original_cols " << original_cols << "\n";
  meta << "presolve_enabled " << (presolve_enabled ? 1 : 0) << "\n";
  meta << "folding_applied " << (folding_applied ? 1 : 0) << "\n";
  meta << "folded_rows " << folded_rows << "\n";
  meta << "folded_cols " << folded_cols << "\n";
  meta << "reduced_rows " << summary.reduced_rows << "\n";
  meta << "reduced_cols " << summary.reduced_cols << "\n";
  meta << "reduced_nnz " << summary.reduced_nnz << "\n";
  meta << "obj_constant " << lp.obj_constant << "\n";
  meta << "objective_shift " << summary.obj_constant_delta << "\n";
  meta << "iterations " << summary.iterations << "\n";

  tools::write_binary_vector(out_dir / "A_rowPtr_i32.bin",
                             tools::copy_to_host(lp.A.rowPtr, static_cast<std::size_t>(lp.A.rows) + 1));
  tools::write_binary_vector(out_dir / "A_colVal_i32.bin",
                             tools::copy_to_host(lp.A.colVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "A_nzVal_f64.bin",
                             tools::copy_to_host(lp.A.nzVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "c_f64.bin",
                             tools::copy_to_host(lp.c, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "l_f64.bin",
                             tools::copy_to_host(lp.l, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "u_f64.bin",
                             tools::copy_to_host(lp.u, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "AL_f64.bin",
                             tools::copy_to_host(lp.AL, static_cast<std::size_t>(lp.A.rows)));
  tools::write_binary_vector(out_dir / "AU_f64.bin",
                             tools::copy_to_host(lp.AU, static_cast<std::size_t>(lp.A.rows)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    gpu_presolver::presolve::PresolveParams params;
    params.enable_folding = true;
    gpu_presolver::tools::apply_presolve_config_override(argc, argv, params);
    apply_env_overrides(params);
    const gpu_presolver::tools::PresolveCliOptions cli =
        gpu_presolver::tools::parse_presolve_cli_options(argc, argv, params);
    if (cli.show_help) {
      std::cout << "usage: gpu_presolver_export_reduced [options] <model.mps|-> <out_dir>\n";
      gpu_presolver::tools::print_presolve_cli_options(std::cout);
      return 0;
    }
    if (cli.positional.size() != 2) {
      std::cerr << "usage: gpu_presolver_export_reduced [options] <model.mps|-> <out_dir>\n";
      return 2;
    }
    if (cli.device >= 0) {
      gpu_presolver::tools::check_cuda(cudaSetDevice(cli.device), "cudaSetDevice");
    }

    const std::string& input_path = cli.positional[0];
    const std::filesystem::path out_dir = cli.positional[1];
    cpu_presolve::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::folding::FoldingPipelineSummary folding_summary;
    gpu_presolver::folding::FoldingRunSummary folding_only_summary;
    gpu_presolver::presolve::GpuPresolveSummary standalone_summary;
    gpu_presolver::presolve::GpuPresolveSummary* summary = nullptr;
    bool folding_applied = false;
    std::int32_t folded_rows = static_cast<std::int32_t>(model.lp.num_rows());
    std::int32_t folded_cols = static_cast<std::int32_t>(model.lp.num_cols());
    if (cli.enable_presolve && params.enable_folding) {
      folding_summary =
          gpu_presolver::folding::run_gpu_presolve_with_folding(device_lp.lp, params, true, true);
      summary = &folding_summary.presolve;
      folding_applied = folding_summary.folding_applied;
      folded_rows = folding_summary.folded_rows;
      folded_cols = folding_summary.folded_cols;
    } else if (cli.enable_presolve) {
      standalone_summary = gpu_presolver::presolve::run_gpu_presolve_with_reduced_lp(device_lp.lp, params);
      summary = &standalone_summary;
    } else if (params.enable_folding) {
      folding_only_summary =
          gpu_presolver::folding::run_folding(device_lp.lp, params.folding_tolerance, params.verbose);
      folding_applied = folding_only_summary.applied;
      folded_rows = folding_only_summary.folded_rows;
      folded_cols = folding_only_summary.folded_cols;
      const gpu_presolver::presolve::LPInfoGpu& output_lp =
          folding_applied ? folding_only_summary.folded_lp : device_lp.lp;
      standalone_summary = gpu_presolver::tools::make_passthrough_summary(output_lp);
      summary = &standalone_summary;
    } else {
      standalone_summary = gpu_presolver::tools::make_passthrough_summary(device_lp.lp);
      summary = &standalone_summary;
    }

    write_lp_export(out_dir,
                    model.name,
                    *summary,
                    static_cast<std::int32_t>(model.lp.num_rows()),
                    static_cast<std::int32_t>(model.lp.num_cols()),
                    cli.enable_presolve,
                    folding_applied,
                    folded_rows,
                    folded_cols);
    std::cout << "status " << (summary->has_infeasible ? "infeasible" : (summary->has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "presolve_enabled " << (cli.enable_presolve ? 1 : 0) << "\n";
    std::cout << "folding_applied " << (folding_applied ? 1 : 0) << "\n";
    std::cout << "folded_rows " << folded_rows << "\n";
    std::cout << "folded_cols " << folded_cols << "\n";
    std::cout << "reduced_rows " << summary->reduced_rows << "\n";
    std::cout << "reduced_cols " << summary->reduced_cols << "\n";
    std::cout << "reduced_nnz " << summary->reduced_nnz << "\n";
    std::cout << "out_dir " << out_dir.string() << "\n";
    const int exit_code = summary->has_infeasible || summary->has_unbounded ? 1 : 0;
    gpu_presolver::presolve::free_gpu_presolve_reduced_lp(*summary);
    if (cli.enable_presolve && params.enable_folding) {
      gpu_presolver::folding::free_folded_lp(folding_summary.folding);
    } else if (!cli.enable_presolve && params.enable_folding) {
      gpu_presolver::folding::free_folded_lp(folding_only_summary);
    }
    return exit_code;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
