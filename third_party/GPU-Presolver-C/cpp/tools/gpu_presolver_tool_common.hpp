#pragma once

#include "cpu_presolve/mpsreader.hpp"
#include "gpu_presolver/presolve/presolve_config.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::tools {

struct PresolveCliOptions {
  bool show_help = false;
  bool enable_presolve = true;
  int device = -1;
  std::string config_path;
  std::vector<std::string> positional;
};

inline double parse_positive_finite_double(const std::string& text, const char* option) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) + " requires a positive finite number");
  }
  if (consumed != text.size() || !std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(option) + " requires a positive finite number");
  }
  return value;
}

inline bool parse_binary_option(const std::string& text, const char* option) {
  if (text == "1") {
    return true;
  }
  if (text == "0") {
    return false;
  }
  throw std::invalid_argument(std::string(option) + " requires 0 or 1");
}

inline int parse_nonnegative_int(const std::string& text, const char* option) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) + " requires a nonnegative integer");
  }
  if (consumed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(option) + " requires a nonnegative integer");
  }
  return static_cast<int>(value);
}

inline bool take_option_value(const std::string& arg,
                              const char* option,
                              int& index,
                              int argc,
                              char** argv,
                              std::string& value) {
  if (arg == option) {
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(option) + " requires a value");
    }
    value = argv[++index];
    return true;
  }
  const std::string prefix = std::string(option) + "=";
  if (arg.rfind(prefix, 0) == 0) {
    value = arg.substr(prefix.size());
    return true;
  }
  return false;
}

inline void print_presolve_cli_options(std::ostream& out) {
  out << "options:\n"
      << "  --presolve <0|1>  enable GPU presolve (default: 1)\n"
      << "  --folding <0|1>   enable folding (default: 1)\n"
      << "  --device <id>      select a CUDA device\n"
      << "  --config <path>    load a TOML config\n"
      << "  -h, --help         show help\n";
}

inline std::string find_presolve_config_path(int argc, char** argv) {
  std::string config_path;
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (take_option_value(argv[i], "--config", i, argc, argv, value)) {
      if (value.empty()) {
        throw std::invalid_argument("--config requires a nonempty path");
      }
      config_path = value;
    }
  }
  return config_path;
}

inline void apply_presolve_config_override(
    int argc,
    char** argv,
    gpu_presolver::presolve::PresolveParams& params) {
  const std::string path = find_presolve_config_path(argc, argv);
  if (!path.empty()) {
    gpu_presolver::presolve::load_presolve_params_from_toml(path, params);
  }
}

inline PresolveCliOptions parse_presolve_cli_options(
    int argc,
    char** argv,
    gpu_presolver::presolve::PresolveParams& params) {
  PresolveCliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string value;
    if (arg == "-h" || arg == "--help") {
      options.show_help = true;
      continue;
    }
    if (take_option_value(arg, "--config", i, argc, argv, value)) {
      if (value.empty()) {
        throw std::invalid_argument("--config requires a nonempty path");
      }
      options.config_path = value;
      continue;
    }
    if (take_option_value(arg, "--device", i, argc, argv, value)) {
      options.device = parse_nonnegative_int(value, "--device");
      continue;
    }
    if (take_option_value(arg, "--presolve", i, argc, argv, value)) {
      options.enable_presolve = parse_binary_option(value, "--presolve");
      continue;
    }
    if (take_option_value(arg, "--folding", i, argc, argv, value)) {
      params.enable_folding = parse_binary_option(value, "--folding");
      continue;
    }
    if (arg != "-" && !arg.empty() && arg.front() == '-') {
      throw std::invalid_argument("unknown option: " + arg);
    }
    options.positional.push_back(arg);
  }
  return options;
}

inline bool parse_bool_env_value(const char* value, bool fallback) {
  if (value == nullptr) {
    return fallback;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (text == "1" || text == "true" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "0" || text == "false" || text == "no" || text == "off") {
    return false;
  }
  return fallback;
}

inline void apply_presolve_env_overrides(gpu_presolver::presolve::PresolveParams& params) {
  if (const char* value = std::getenv("GPUPRESOLVER_MAX_ITERS")) {
    params.max_iters = std::max(0, std::atoi(value));
  }
  if (const char* value = std::getenv("GPUPRESOLVER_SCHEDULER")) {
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (text == "fixed" || text == "legacy" || text == "old") {
      params.use_tiered_scheduler = false;
    } else if (text == "tiered" || text == "default") {
      params.use_tiered_scheduler = true;
    }
  }
  params.use_tiered_scheduler = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_USE_TIERED_SCHEDULER"), params.use_tiered_scheduler);
  params.enable_doubleton_eq = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_DOUBLETON_EQ"), params.enable_doubleton_eq);
  params.enable_redundant_bounds = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_REDUNDANT_BOUNDS"), params.enable_redundant_bounds);
  params.enable_structural_l1_substitution = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_STRUCTURAL_L1_SUBSTITUTION"),
      params.enable_structural_l1_substitution);
  params.enable_folding = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_FOLDING"), params.enable_folding);
  if (const char* value = std::getenv("GPUPRESOLVER_FOLDING_TOLERANCE")) {
    params.folding_tolerance = std::atof(value);
  }
}

inline gpu_presolver::presolve::GpuPresolveSummary make_passthrough_summary(
    const gpu_presolver::presolve::LPInfoGpu& lp) {
  gpu_presolver::presolve::GpuPresolveSummary summary;
  summary.original_rows = lp.A.rows;
  summary.original_cols = lp.A.cols;
  summary.reduced_rows = lp.A.rows;
  summary.reduced_cols = lp.A.cols;
  summary.reduced_nnz = lp.A.nnz;
  summary.reduced_lp = lp;
  return summary;
}

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check_cuda(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(const T* device, std::size_t size) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  check_cuda(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  return values;
}

inline std::vector<std::int32_t> to_i32(const std::vector<int>& input) {
  std::vector<std::int32_t> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<std::int32_t>(input[i]);
  }
  return out;
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

inline DeviceLpOwner upload_lp(const cpu_presolve::LpModel& model) {
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

inline cpu_presolve::MpsModel read_mps_file(const std::string& path) {
  std::unique_ptr<std::ifstream> file;
  std::istream* input = &std::cin;
  if (path != "-") {
    file = std::make_unique<std::ifstream>(path);
    if (!file->is_open()) {
      throw std::runtime_error("failed to open " + path);
    }
    input = file.get();
  }
  return cpu_presolve::read_mps(*input);
}

template <class T>
void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (!values.empty()) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(sizeof(T) * values.size()));
  }
}

template <class T>
std::vector<T> read_binary_vector(const std::filesystem::path& path, std::size_t count) {
  std::vector<T> values(count);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (count > 0) {
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
      throw std::runtime_error("failed to read " + path.string());
    }
  }
  return values;
}

}  // namespace gpu_presolver::tools
