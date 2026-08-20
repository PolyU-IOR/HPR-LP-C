#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void apply_env_overrides(gpu_presolver::presolve::PresolveParams& params) {
  gpu_presolver::tools::apply_presolve_env_overrides(params);
}

double finite_bound_norm_l2(const std::vector<double>& lower,
                            const std::vector<double>& upper) {
  double sum_sq = 0.0;
  for (const double v : lower) {
    if (std::isfinite(v)) {
      sum_sq += v * v;
    }
  }
  for (const double v : upper) {
    if (std::isfinite(v)) {
      sum_sq += v * v;
    }
  }
  return 1.0 + std::sqrt(sum_sq);
}

struct PrimalFeasibility {
  double normalized = 0.0;
  double row_l2 = 0.0;
  double row_max = 0.0;
  double col_l2 = 0.0;
  double col_max = 0.0;
};

struct ViolationDetail {
  int index = -1;
  double violation = 0.0;
  double value = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

void keep_top_violation(std::vector<ViolationDetail>& top, ViolationDetail item, std::size_t limit) {
  if (item.violation <= 0.0 || limit == 0) {
    return;
  }
  top.push_back(item);
  std::sort(top.begin(), top.end(), [](const ViolationDetail& lhs, const ViolationDetail& rhs) {
    if (lhs.violation != rhs.violation) {
      return lhs.violation > rhs.violation;
    }
    return lhs.index < rhs.index;
  });
  if (top.size() > limit) {
    top.resize(limit);
  }
}

PrimalFeasibility compute_primal_feasibility(const cpu_presolve::LpModel& model,
                                             const std::vector<double>& x) {
  const cpu_presolve::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  PrimalFeasibility out;
  double row_sq = 0.0;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (int p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    row_sq += violation * violation;
    out.row_max = std::max(out.row_max, violation);
  }
  out.row_l2 = std::sqrt(row_sq);

  double col_sq = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    col_sq += violation * violation;
    out.col_max = std::max(out.col_max, violation);
  }
  out.col_l2 = std::sqrt(col_sq);
  out.normalized = std::max(out.row_l2, out.col_l2) / finite_bound_norm_l2(AL, AU);
  return out;
}

void print_top_violations(const cpu_presolve::LpModel& model,
                          const std::vector<double>& x,
                          std::size_t limit) {
  const cpu_presolve::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  std::vector<ViolationDetail> top_rows;
  std::vector<ViolationDetail> top_cols;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (int p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    keep_top_violation(top_rows,
                       ViolationDetail{i, violation, activity, AL[static_cast<std::size_t>(i)],
                                       AU[static_cast<std::size_t>(i)]},
                       limit);
  }

  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    keep_top_violation(top_cols,
                       ViolationDetail{static_cast<int>(j), violation, x[j], l[j], u[j]},
                       limit);
  }

  for (std::size_t k = 0; k < top_rows.size(); ++k) {
    const ViolationDetail& item = top_rows[k];
    std::cout << "top_row_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
  for (std::size_t k = 0; k < top_cols.size(); ++k) {
    const ViolationDetail& item = top_cols[k];
    std::cout << "top_col_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
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
      std::cout << "usage: gpu_presolver_check_postsolve_solution [options] "
                   "<model.mps|-> <solution_dir>\n";
      gpu_presolver::tools::print_presolve_cli_options(std::cout);
      return 0;
    }
    if (cli.positional.size() != 2) {
      std::cerr << "usage: gpu_presolver_check_postsolve_solution [options] "
                   "<model.mps|-> <solution_dir>\n";
      return 2;
    }
    if (cli.device >= 0) {
      gpu_presolver::tools::check_cuda(cudaSetDevice(cli.device), "cudaSetDevice");
    }

    const std::string& input_path = cli.positional[0];
    const std::filesystem::path solution_dir = cli.positional[1];
    cpu_presolve::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::folding::FoldingPipelineSummary folding_summary;
    gpu_presolver::folding::FoldingRunSummary folding_only_summary;
    gpu_presolver::presolve::GpuPresolveSummary standalone_summary;
    gpu_presolver::presolve::GpuPresolveSummary* summary = nullptr;
    bool folding_applied = false;
    std::int32_t folded_rows = device_lp.lp.A.rows;
    std::int32_t folded_cols = device_lp.lp.A.cols;
    if (cli.enable_presolve && params.enable_folding) {
      folding_summary =
          gpu_presolver::folding::run_gpu_presolve_with_folding(device_lp.lp, params, false, true);
      summary = &folding_summary.presolve;
      folding_applied = folding_summary.folding_applied;
      folded_rows = folding_summary.folded_rows;
      folded_cols = folding_summary.folded_cols;
    } else if (cli.enable_presolve) {
      standalone_summary = gpu_presolver::presolve::run_gpu_presolve_with_record(device_lp.lp, params);
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

    const std::vector<double> x_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "x_f64.bin",
                                                         static_cast<std::size_t>(summary->reduced_cols));
    const std::vector<double> y_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "y_f64.bin",
                                                         static_cast<std::size_t>(summary->reduced_rows));
    const std::vector<double> z_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "z_f64.bin",
                                                         static_cast<std::size_t>(summary->reduced_cols));

    std::vector<double> x_org;
    if (!cli.enable_presolve) {
      if (folding_applied) {
        gpu_presolver::folding::UnfoldedSolutionHost unfolded =
            gpu_presolver::folding::unfold_solution_to_host(
                folding_only_summary.map, x_red_h, y_red_h, z_red_h);
        x_org = std::move(unfolded.x);
      } else {
        x_org = x_red_h;
      }
    } else {
      double* x_red = gpu_presolver::tools::copy_to_device(x_red_h);
      double* y_red = gpu_presolver::tools::copy_to_device(y_red_h);
      double* z_red = gpu_presolver::tools::copy_to_device(z_red_h);
      if (params.enable_folding) {
        gpu_presolver::folding::UnfoldedSolutionHost unfolded =
            gpu_presolver::folding::postsolve_and_unfold_to_host(x_red, y_red, z_red, folding_summary);
        x_org = std::move(unfolded.x);
      } else {
        gpu_presolver::presolve::GpuPostsolveResult post =
            gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, summary->record, &device_lp.lp);
        x_org = gpu_presolver::tools::copy_to_host(
            post.x_org, static_cast<std::size_t>(summary->original_cols));
        cudaFree(post.x_org);
        cudaFree(post.y_org);
        cudaFree(post.z_org);
      }
      cudaFree(x_red);
      cudaFree(y_red);
      cudaFree(z_red);
    }
    const PrimalFeasibility feas = compute_primal_feasibility(model.lp, x_org);

    std::cout << "status " << (summary->has_infeasible ? "infeasible" : (summary->has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "presolve_enabled " << (cli.enable_presolve ? 1 : 0) << "\n";
    std::cout << "folding_applied " << (folding_applied ? 1 : 0) << "\n";
    std::cout << "folded_rows " << folded_rows << "\n";
    std::cout << "folded_cols " << folded_cols << "\n";
    std::cout << "reduced_rows " << summary->reduced_rows << "\n";
    std::cout << "reduced_cols " << summary->reduced_cols << "\n";
    std::cout << "primal_feas_l2_normalized " << feas.normalized << "\n";
    std::cout << "row_violation_l2 " << feas.row_l2 << "\n";
    std::cout << "row_violation_max " << feas.row_max << "\n";
    std::cout << "col_violation_l2 " << feas.col_l2 << "\n";
    std::cout << "col_violation_max " << feas.col_max << "\n";
    if (gpu_presolver::tools::parse_bool_env_value(
            std::getenv("GPUPRESOLVER_PRINT_TOP_VIOLATIONS"), false)) {
      print_top_violations(model.lp, x_org, 10);
    }

    if (cli.enable_presolve && params.enable_folding) {
      gpu_presolver::folding::free_folded_lp(folding_summary.folding);
    } else if (!cli.enable_presolve && params.enable_folding) {
      gpu_presolver::folding::free_folded_lp(folding_only_summary);
    }
    return feas.normalized <= 1.0e-6 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
