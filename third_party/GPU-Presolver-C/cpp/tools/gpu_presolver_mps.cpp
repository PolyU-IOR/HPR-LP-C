#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "cpu_presolve/mpsreader.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

std::vector<std::int32_t> to_i32(const std::vector<int>& input) {
  std::vector<std::int32_t> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<std::int32_t>(input[i]);
  }
  return out;
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

struct DeviceLpOwner {
  gpu_presolver::presolve::LPInfoGpu lp;

  ~DeviceLpOwner() {
    cudaFree(lp.A.rowPtr);
    cudaFree(lp.A.colVal);
    cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);
    cudaFree(lp.AT.colVal);
    cudaFree(lp.AT.nzVal);
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
  }
};

DeviceLpOwner upload_lp(const cpu_presolve::LpModel& model) {
  DeviceLpOwner owner;
  const cpu_presolve::CsrMatrix& csr = model.csr();
  const cpu_presolve::CscMatrix& csc = model.csc();

  const std::vector<std::int32_t> row_ptr = to_i32(csr.row_ptr());
  const std::vector<std::int32_t> col_idx = to_i32(csr.col_idx());
  const std::vector<std::int32_t> at_row_ptr = to_i32(csc.col_ptr());
  const std::vector<std::int32_t> at_col_idx = to_i32(csc.row_idx());

  owner.lp.A.rows = static_cast<std::int32_t>(csr.rows());
  owner.lp.A.cols = static_cast<std::int32_t>(csr.cols());
  owner.lp.A.nnz = static_cast<std::int32_t>(csr.nnz());
  owner.lp.A.rowPtr = copy_to_device(row_ptr);
  owner.lp.A.colVal = copy_to_device(col_idx);
  owner.lp.A.nzVal = copy_to_device(csr.values());

  owner.lp.AT.rows = static_cast<std::int32_t>(csc.cols());
  owner.lp.AT.cols = static_cast<std::int32_t>(csc.rows());
  owner.lp.AT.nnz = static_cast<std::int32_t>(csc.nnz());
  owner.lp.AT.rowPtr = copy_to_device(at_row_ptr);
  owner.lp.AT.colVal = copy_to_device(at_col_idx);
  owner.lp.AT.nzVal = copy_to_device(csc.values());

  owner.lp.c = copy_to_device(model.objective());
  owner.lp.AL = copy_to_device(model.row_lower());
  owner.lp.AU = copy_to_device(model.row_upper());
  owner.lp.l = copy_to_device(model.col_lower());
  owner.lp.u = copy_to_device(model.col_upper());
  owner.lp.obj_constant = model.obj_constant();
  return owner;
}

const char* status_name(const gpu_presolver::presolve::GpuPresolveSummary& summary) {
  if (summary.has_infeasible) {
    return "infeasible";
  }
  if (summary.has_unbounded) {
    return "unbounded";
  }
  return "ok";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    gpu_presolver::presolve::PresolveParams params;
    params.enable_folding = true;
    gpu_presolver::tools::apply_presolve_config_override(argc, argv, params);
    gpu_presolver::tools::apply_presolve_env_overrides(params);
    const gpu_presolver::tools::PresolveCliOptions cli =
        gpu_presolver::tools::parse_presolve_cli_options(argc, argv, params);
    if (cli.show_help) {
      std::cout << "usage: gpu_presolver_mps [options] <model.mps|->\n";
      gpu_presolver::tools::print_presolve_cli_options(std::cout);
      return 0;
    }
    if (cli.positional.size() != 1) {
      std::cerr << "usage: gpu_presolver_mps [options] <model.mps|->\n";
      return 2;
    }
    if (cli.device >= 0) {
      check(cudaSetDevice(cli.device), "cudaSetDevice");
    }

    std::unique_ptr<std::ifstream> file;
    std::istream* input = &std::cin;
    const std::string& path = cli.positional[0];
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

    DeviceLpOwner device_lp = upload_lp(model.lp);
    const auto presolve_start = std::chrono::steady_clock::now();
    gpu_presolver::folding::FoldingPipelineSummary folding_summary;
    gpu_presolver::folding::FoldingRunSummary folding_only_summary;
    gpu_presolver::presolve::GpuPresolveSummary standalone_summary;
    gpu_presolver::presolve::GpuPresolveSummary* summary = nullptr;
    bool folding_applied = false;
    std::int32_t folded_rows = device_lp.lp.A.rows;
    std::int32_t folded_cols = device_lp.lp.A.cols;
    if (cli.enable_presolve && params.enable_folding) {
      folding_summary =
          gpu_presolver::folding::run_gpu_presolve_with_folding(device_lp.lp, params, false, false);
      summary = &folding_summary.presolve;
      folding_applied = folding_summary.folding_applied;
      folded_rows = folding_summary.folded_rows;
      folded_cols = folding_summary.folded_cols;
    } else if (cli.enable_presolve) {
      standalone_summary = gpu_presolver::presolve::run_gpu_presolve_fixed_order(device_lp.lp, params);
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
    const auto presolve_stop = std::chrono::steady_clock::now();
    const std::int32_t original_rows = device_lp.lp.A.rows;
    const std::int32_t original_cols = device_lp.lp.A.cols;

    std::cout << "model " << model.name << '\n';
    std::cout << "status " << status_name(*summary) << '\n';
    std::cout << "presolve_enabled " << (cli.enable_presolve ? 1 : 0) << '\n';
    std::cout << "changed " << ((summary->reduced_rows != original_rows ||
                                  summary->reduced_cols != original_cols) ? 1 : 0) << '\n';
    std::cout << "original_rows " << original_rows << '\n';
    std::cout << "original_cols " << original_cols << '\n';
    std::cout << "original_nnz " << device_lp.lp.A.nnz << '\n';
    std::cout << "folding_applied " << (folding_applied ? 1 : 0) << '\n';
    std::cout << "folded_rows " << folded_rows << '\n';
    std::cout << "folded_cols " << folded_cols << '\n';
    std::cout << "reduced_rows " << summary->reduced_rows << '\n';
    std::cout << "reduced_cols " << summary->reduced_cols << '\n';
    std::cout << "reduced_nnz " << summary->reduced_nnz << '\n';
    std::cout << "fixed_columns 0\n";
    std::cout << "objective_shift " << summary->obj_constant_delta << '\n';
    std::cout << "iterations " << summary->iterations << '\n';
    std::cout << "parse_seconds " << seconds_since(parse_start, parse_stop) << '\n';
    std::cout << "presolve_seconds " << seconds_since(presolve_start, presolve_stop) << '\n';
    const int exit_code = summary->has_infeasible || summary->has_unbounded ? 1 : 0;
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
