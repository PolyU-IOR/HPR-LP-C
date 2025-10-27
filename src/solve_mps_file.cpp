#include <iostream>
#include <string>
#include <cstring>
#include <sys/stat.h>  // For file existence check (C++11 compatible)
#include "mps_reader.h"
#include "HPRLP.h"

// C++11 compatible file existence check
inline bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i <input.mps> [options]\n"
              << "\nOptions:\n"
              << "  -i, --input <path>         Path to input .mps file (required)\n"
              << "      --device <id>          CUDA device id (default: 0)\n"
              << "      --max-iter <N>         Max iterations (default: INT32_MAX)\n"
              << "      --tol <eps>            Stopping tolerance (default: 1e-4)\n"
              << "      --time-limit <sec>     Time limit in seconds (default: 3600)\n"
              << "      --check-iter <N>       Check interval (default: 150)\n"
              << "      --ruiz <true/false>    Enable/disable Ruiz scaling (default: true)\n"
              << "      --pock <true/false>    Enable/disable Pock-Chambolle scaling (default: true)\n"
              << "      --bc <true/false>      Enable/disable bounds/cost scaling (default: true)\n"
              << "  -h, --help                 Show this help and exit\n"
              << "\nExample:\n  " << prog << " -i model.mps --device 0 --time-limit 3600 --tol 1e-4\n";
}

int main(int argc, char** argv) {
    std::string input_path;
    bool input_provided = false;
    HPRLP_parameters param; // defaults from structs.h

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
        } else if (std::strcmp(a, "--max-iter") == 0) {
            need_value(a);
            param.max_iter = std::stoi(argv[++i]);
        } else if (std::strcmp(a, "--tol") == 0) {
            need_value(a);
            param.stop_tol = static_cast<HPRLP_FLOAT>(std::stod(argv[++i]));
        } else if (std::strcmp(a, "--time-limit") == 0) {
            need_value(a);
            param.time_limit = std::stod(argv[++i]);
        } else if (std::strcmp(a, "--check-iter") == 0) {
            need_value(a);
            param.check_iter = std::stoi(argv[++i]);
        } else if (std::strcmp(a, "--ruiz") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_Ruiz_scaling = (val == "true" || val == "1");
        } else if (std::strcmp(a, "--pock") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_Pock_Chambolle_scaling = (val == "true" || val == "1");
        } else if (std::strcmp(a, "--bc") == 0) {
            need_value(a);
            std::string val = argv[++i];
            param.use_bc_scaling = (val == "true" || val == "1");
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

    if (!file_exists(input_path)) {
        std::cerr << "Input file does not exist: " << input_path << "\n";
        print_usage(argv[0]);
        return 1;
    }

    LP_info_cpu lp_info;
    build_model_from_mps(input_path.c_str(), &lp_info);

    HPRLP_results output = HPRLP_main_solve(&lp_info, &param);
    
    return 0;
}