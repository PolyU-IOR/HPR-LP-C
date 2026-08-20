#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <sys/stat.h>  // For file existence check (C++11 compatible)
#include "io/mps_reader.h"
#include "HPRLP.h"
#include "solver/internal/solver_output.h"

#ifdef HPRLP_HAS_HDF5
#include <hdf5.h>
#endif

// C++11 compatible file existence check
inline bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static void apply_default_runtime_environment() {
    static const struct {
        const char* name;
        const char* value;
    } defaults[] = {
        {"HPRLP_ENABLE_COMPRESSIBLE_MEMORY", "1"},
        {"HPRLP_USE_ROW_REDUCTION", "1"},
        {"HPRLP_USE_ROW_COMPRESSED_AUTOTUNE", "1"},
        {"HPRLP_USE_REDUCED_COMPRESSED_AUTOTUNE", "1"},
        {"HPRLP_DEFER_REDUCED_EMPTY_ROWS_TO_CHECK", "1"},
        {"HPRLP_USE_REDUCED_NONEMPTY_CUSPARSE", "1"},
        {"HPRLP_REDUCED_RESET_MASK_ON_RESTART", "1"},
        {"HPRLP_REDUCED_RESTART_MASK_MIN_RECOVERY", "0.25"},
        {"HPRLP_REDUCED_RESTART_MASK_MIN_SAVED_COLUMNS", "25000"},
        {"HPRLP_REDUCED_RESTART_MASK_MIN_CURRENT_COLUMNS", "95000"},
    };
    for (const auto& setting : defaults) {
        if (std::getenv(setting.name) == nullptr) {
            ::setenv(setting.name, setting.value, 0);
        }
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i <input.mps|input.mps.gz|input.h5> [options]\n"
              << "\nOptions:\n"
              << "  -i, --input <path>         Path to input .mps, .mps.gz, .h5, or .hdf5 file (required)\n"
              << "      --device <id>          CUDA device id (default: 0)\n"
              << "      --max-iter <N>         Max iterations (default: INT32_MAX)\n"
              << "      --tol <eps>            Stopping tolerance (default: 1e-6)\n"
              << "      --time-limit <sec>     Time limit in seconds (default: 1000)\n"
              << "      --check-iter <N>       Check interval (default: 150)\n"
              << "      --cusparse-spmv <true/false>  Force cuSPARSE normal updates (default: false)\n"
              << "      --autotune-verbose <true/false>  Print fused backend autotune results (default: false)\n"
              << "      --progress-monitor <true/false>  Sample active-set/direction progress (default: true)\n"
              << "      --progress-control <true/false>  Enable sigma safeguards and phase state (default: true)\n"
              << "      --sigma-rebalance <true/false>  Enable closed-loop flag-4 trials (default: true)\n"
              << "      --restart-guard <true/false>  Enable long-restart deferral (default: false)\n"
              << "      --restart-cooldown <checks>  Ordinary restart cooldown (default: 0)\n"
              << "      --debug-restart <true/false>  Print restart decisions (default: false)\n"
              << "      --debug-sigma <true/false>  Print sigma decisions (default: false)\n"
              << "      --print-debug-info <true/false>  Print detailed solver diagnostics (default: false)\n"
              << "      --fixed-sigma <value>  Use a fixed positive sigma (default: adaptive)\n"
              << "      --cr <true/false>      Enable/disable Curtis-Reid prescaling (default: true)\n"
              << "      --ruiz <true/false>    Enable/disable Ruiz scaling (default: true)\n"
              << "      --pock <true/false>    Enable/disable Pock-Chambolle scaling (default: true)\n"
              << "      --bc <true/false>      Enable/disable bounds/cost scaling (default: true)\n"
              << "      --presolver <pslp|gpu|none>  Select presolver backend (default: gpu)\n"
              << "      --gpu-folding <true/false>  Enable/disable GPU presolver folding (default: true)\n"
              << "      --reduced-matrix <true/false>  Enable adaptive row/column reduction in manual mode (default: true)\n"
              << "      --auto-memory-policy <true/false>  Couple reduced/compression from presolved dimensions (default: false)\n"
              << "  -h, --help                 Show this help and exit\n"
              << "\nExample:\n  " << prog << " -i model.mps.gz --device 0 --time-limit 1000 --tol 1e-6\n";
}

static bool ends_with_case_insensitive(const std::string& value,
                                       const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return std::equal(
        suffix.rbegin(), suffix.rend(), value.rbegin(),
        [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) ==
                   std::tolower(static_cast<unsigned char>(right));
        });
}

static bool is_hdf5_path(const std::string& path) {
    return ends_with_case_insensitive(path, ".h5") ||
           ends_with_case_insensitive(path, ".hdf5");
}

static bool is_mps_path(const std::string& path) {
    return ends_with_case_insensitive(path, ".mps") ||
           ends_with_case_insensitive(path, ".mps.gz");
}

#ifdef HPRLP_HAS_HDF5
template <typename T>
static std::vector<T> read_hdf5_dataset(
    hid_t file, const char* path, hid_t memory_type) {
    hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
    if (dataset < 0) {
        throw std::runtime_error(std::string("Missing HDF5 dataset: ") + path);
    }
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0) {
        H5Dclose(dataset);
        throw std::runtime_error(
            std::string("Cannot inspect HDF5 dataset: ") + path);
    }

    const int rank = H5Sget_simple_extent_ndims(dataspace);
    if (rank < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error(
            std::string("Cannot read HDF5 dataset rank: ") + path);
    }
    std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
    if (rank > 0 &&
        H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr) < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error(
            std::string("Cannot read HDF5 dataset dimensions: ") + path);
    }

    std::size_t count = 1;
    for (int index = 0; index < rank; ++index) {
        const hsize_t dimension = dimensions[static_cast<std::size_t>(index)];
        if (dimension == 0 || count == 0) {
            count = 0;
            continue;
        }
        if (dimension > std::numeric_limits<std::size_t>::max() / count) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
            throw std::runtime_error(
                std::string("HDF5 dataset is too large: ") + path);
        }
        count *= static_cast<std::size_t>(dimension);
    }

    std::vector<T> values(count);
    if (count > 0 &&
        H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                values.data()) < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        throw std::runtime_error(
            std::string("Cannot read HDF5 dataset: ") + path);
    }
    H5Sclose(dataspace);
    H5Dclose(dataset);
    return values;
}

template <typename T>
static T read_hdf5_scalar(hid_t file, const char* path, hid_t memory_type) {
    const std::vector<T> values =
        read_hdf5_dataset<T>(file, path, memory_type);
    if (values.size() != 1) {
        throw std::runtime_error(
            std::string("Expected scalar HDF5 dataset: ") + path);
    }
    return values[0];
}

static LP_info_cpu* create_model_from_hdf5_file(const std::string& path) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    if (H5Fis_hdf5(path.c_str()) <= 0) {
        std::cerr << "Input has an HDF5 extension but is not a valid HDF5 file: "
                  << path << "\n";
        return nullptr;
    }

    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Failed to open HDF5 file: " << path << "\n";
        return nullptr;
    }

    LP_info_cpu* model = nullptr;
    try {
        const std::int32_t schema_version = read_hdf5_scalar<std::int32_t>(
            file, "schema_version", H5T_NATIVE_INT32);
        if (schema_version != 1) {
            throw std::runtime_error(
                "Unsupported LP HDF5 schema version " +
                std::to_string(schema_version));
        }

        const std::vector<std::int64_t> matrix_size =
            read_hdf5_dataset<std::int64_t>(
                file, "A/size", H5T_NATIVE_INT64);
        if (matrix_size.size() != 2 ||
            matrix_size[0] <= 0 || matrix_size[1] <= 0 ||
            matrix_size[0] > std::numeric_limits<int>::max() ||
            matrix_size[1] > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Invalid A/size dataset");
        }
        const int m = static_cast<int>(matrix_size[0]);
        const int n = static_cast<int>(matrix_size[1]);

        std::vector<int> colptr =
            read_hdf5_dataset<int>(file, "A/colptr", H5T_NATIVE_INT);
        std::vector<int> rowval =
            read_hdf5_dataset<int>(file, "A/rowval", H5T_NATIVE_INT);
        const std::vector<HPRLP_FLOAT> nzval =
            read_hdf5_dataset<HPRLP_FLOAT>(
                file, "A/nzval", H5T_NATIVE_DOUBLE);
        const std::vector<HPRLP_FLOAT> c =
            read_hdf5_dataset<HPRLP_FLOAT>(file, "c", H5T_NATIVE_DOUBLE);
        const std::vector<HPRLP_FLOAT> AL =
            read_hdf5_dataset<HPRLP_FLOAT>(file, "AL", H5T_NATIVE_DOUBLE);
        const std::vector<HPRLP_FLOAT> AU =
            read_hdf5_dataset<HPRLP_FLOAT>(file, "AU", H5T_NATIVE_DOUBLE);
        const std::vector<HPRLP_FLOAT> l =
            read_hdf5_dataset<HPRLP_FLOAT>(file, "l", H5T_NATIVE_DOUBLE);
        const std::vector<HPRLP_FLOAT> u =
            read_hdf5_dataset<HPRLP_FLOAT>(file, "u", H5T_NATIVE_DOUBLE);
        const HPRLP_FLOAT obj_constant = read_hdf5_scalar<HPRLP_FLOAT>(
            file, "obj_constant", H5T_NATIVE_DOUBLE);

        if (rowval.size() != nzval.size() ||
            rowval.size() >=
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Invalid HDF5 sparse value arrays");
        }
        const int nnz = static_cast<int>(rowval.size());
        if (colptr.size() != static_cast<std::size_t>(n) + 1 ||
            c.size() != static_cast<std::size_t>(n) ||
            l.size() != static_cast<std::size_t>(n) ||
            u.size() != static_cast<std::size_t>(n) ||
            AL.size() != static_cast<std::size_t>(m) ||
            AU.size() != static_cast<std::size_t>(m)) {
            throw std::runtime_error("Invalid HDF5 LP vector dimensions");
        }
        if (colptr.front() != 1 || colptr.back() != nnz + 1) {
            throw std::runtime_error(
                "HDF5 A/colptr must use valid 1-based CSC indexing");
        }
        for (std::size_t index = 1; index < colptr.size(); ++index) {
            if (colptr[index] < colptr[index - 1] ||
                colptr[index] < 1 || colptr[index] > nnz + 1) {
                throw std::runtime_error(
                    "HDF5 A/colptr is not monotone or is out of range");
            }
        }
        for (std::size_t index = 0; index < rowval.size(); ++index) {
            if (rowval[index] < 1 || rowval[index] > m) {
                throw std::runtime_error(
                    "HDF5 A/rowval contains an out-of-range row");
            }
            --rowval[index];
        }
        for (std::size_t index = 0; index < colptr.size(); ++index) {
            --colptr[index];
        }

        model = create_model_from_arrays_with_obj_constant(
            m, n, nnz, colptr.data(), rowval.data(), nzval.data(),
            AL.data(), AU.data(), l.data(), u.data(), c.data(),
            obj_constant, true);
        if (!model) {
            throw std::runtime_error(
                "C model construction from HDF5 arrays failed");
        }
    } catch (const std::exception& error) {
        std::cerr << "Failed to load HDF5 model: " << error.what() << "\n";
    }
    H5Fclose(file);

    return model;
}
#endif

int main(int argc, char** argv) {
    apply_default_runtime_environment();
    std::string input_path;
    bool input_provided = false;
    HPRLP_parameters param; // defaults from structs.h
    auto mark_specified = [&](std::uint64_t bit) {
        param.specified_parameter_mask |= bit;
    };

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need_value = [&](const char* opt) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for option: " << opt << "\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        };

        if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(a, "-i") == 0 || std::strcmp(a, "--input") == 0) {
            need_value(a);
            input_path = std::string(argv[++i]);
            input_provided = true;
        } else if (std::strcmp(a, "--device") == 0) {
            need_value(a);
            param.device_number = std::stoi(argv[++i]);
            mark_specified(HPRLP_PARAM_DEVICE);
        } else if (std::strcmp(a, "--max-iter") == 0) {
            need_value(a);
            param.max_iter = std::stoi(argv[++i]);
            mark_specified(HPRLP_PARAM_MAX_ITER);
        } else if (std::strcmp(a, "--tol") == 0) {
            need_value(a);
            param.stop_tol = static_cast<HPRLP_FLOAT>(std::stod(argv[++i]));
            mark_specified(HPRLP_PARAM_STOP_TOL);
        } else if (std::strcmp(a, "--time-limit") == 0) {
            need_value(a);
            param.time_limit = std::stod(argv[++i]);
            mark_specified(HPRLP_PARAM_TIME_LIMIT);
        } else if (std::strcmp(a, "--check-iter") == 0) {
            need_value(a);
            param.check_iter = std::stoi(argv[++i]);
            mark_specified(HPRLP_PARAM_CHECK_ITER);
        } else if (std::strcmp(a, "--cusparse-spmv") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.CUSPARSE_spmv = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_CUSPARSE_SPMV);
        } else if (std::strcmp(a, "--autotune-verbose") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.autotune_verbose = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_AUTOTUNE_VERBOSE);
        } else if (std::strcmp(a, "--progress-monitor") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.enable_progress_monitor = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_PROGRESS_MONITOR);
        } else if (std::strcmp(a, "--progress-control") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.enable_progress_control = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_PROGRESS_CONTROL);
        } else if (std::strcmp(a, "--sigma-rebalance") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.enable_sigma_rebalance_restart =
                (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_SIGMA_REBALANCE);
        } else if (std::strcmp(a, "--restart-guard") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_progress_restart_guard = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_RESTART_GUARD);
        } else if (std::strcmp(a, "--restart-cooldown") == 0) {
            need_value(a);
            param.restart_cooldown_checks = std::stoi(argv[++i]);
            mark_specified(HPRLP_PARAM_RESTART_COOLDOWN);
        } else if (std::strcmp(a, "--debug-restart") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.debug_restart = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_DEBUG_RESTART);
        } else if (std::strcmp(a, "--debug-sigma") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.debug_sigma = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_DEBUG_SIGMA);
        } else if (std::strcmp(a, "--print-debug-info") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.print_debug_info = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_PRINT_DEBUG_INFO);
        } else if (std::strcmp(a, "--fixed-sigma") == 0) {
            need_value(a);
            param.fixed_sigma = static_cast<HPRLP_FLOAT>(std::stod(argv[++i]));
            mark_specified(HPRLP_PARAM_FIXED_SIGMA);
        } else if (std::strcmp(a, "--cr") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_CR_scaling = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_CR_SCALING);
        } else if (std::strcmp(a, "--ruiz") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_Ruiz_scaling = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_RUIZ_SCALING);
        } else if (std::strcmp(a, "--pock") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_Pock_Chambolle_scaling = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_POCK_SCALING);
        } else if (std::strcmp(a, "--bc") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_bc_scaling = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_BC_SCALING);
        } else if (std::strcmp(a, "--presolver") == 0) {
            need_value(a);
            std::string val = argv[++i];
            if (val == "pslp") {
                param.use_presolve = true;
                param.presolver = HPRLP_PRESOLVER_PSLP;
            } else if (val == "gpu") {
                param.use_presolve = true;
                param.presolver = HPRLP_PRESOLVER_GPU;
            } else if (val == "none") {
                param.use_presolve = false;
                param.presolver = HPRLP_PRESOLVER_NONE;
            } else {
                std::cerr << "Unknown presolver backend: " << val << "\n";
                print_usage(argv[0]);
                return 1;
            }
            mark_specified(HPRLP_PARAM_PRESOLVER);
        } else if (std::strcmp(a, "--gpu-folding") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.enable_gpu_folding = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_GPU_FOLDING);
        } else if (std::strcmp(a, "--reduced-matrix") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_reduced_matrix = (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_REDUCED_MATRIX);
        } else if (std::strcmp(a, "--auto-memory-policy") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.auto_reduced_compression_policy =
                (val == "true" || val == "1");
            mark_specified(HPRLP_PARAM_AUTO_MEMORY_POLICY);
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Check if input file is provided
    if (!input_provided) {
        std::cerr << "Error: Input file is required. Use -i or --input option.\n";
        print_usage(argv[0]);
        return 1;
    }

    ::setenv("HPRLP_PRINT_DEBUG_INFO",
             param.print_debug_info ? "1" : "0", 1);

    if (!file_exists(input_path)) {
        std::cerr << "Input file does not exist: " << input_path << "\n";
        print_usage(argv[0]);
        return 1;
    }

    HPRLP_scoped_output_filter input_output_filter(param.print_debug_info);
    std::cout << "Reading file " << input_path << "\n";
    const auto read_start = std::chrono::steady_clock::now();
    LP_info_cpu* model = nullptr;
    if (is_hdf5_path(input_path)) {
#ifdef HPRLP_HAS_HDF5
        model = create_model_from_hdf5_file(input_path);
#else
        std::cerr
            << "This solve_mps_file build has no native HDF5 support. "
               "Rebuild after instantiating bindings/julia/package, or use "
               "scripts/run_dataset.jl.\n";
        return 1;
#endif
    } else if (is_mps_path(input_path)) {
        model = create_model_from_mps(input_path.c_str());
    } else {
        std::cerr << "Unsupported input extension: " << input_path
                  << ". Expected .mps, .mps.gz, .h5, or .hdf5.\n";
        return 1;
    }
    const std::chrono::duration<double> read_elapsed =
        std::chrono::steady_clock::now() - read_start;
    std::cout << "Reading time: " << std::fixed << std::setprecision(2)
              << read_elapsed.count() << "s\n" << std::defaultfloat;
    if (!model) {
        std::cerr << "Failed to load model from input file: "
                  << input_path << "\n";
        return 1;
    }

    HPRLP_results output = solve(model, &param);


    if (output.x) free(output.x);
    if (output.y) free(output.y);
    if (output.z) free(output.z);
    free_model(model);

    return 0;
}
