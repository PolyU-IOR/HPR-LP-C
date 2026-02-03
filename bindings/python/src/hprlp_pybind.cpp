/**
 * @file hprlp_pybind.cpp
 * @brief Pybind11 bindings for HPRLP library
 * 
 * This file provides Python bindings for the HPRLP C/C++ library using pybind11.
 * It exposes the main solver functions and data structures to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>

// Include HPRLP headers (now using proper include paths from CMake)
#include "HPRLP.h"
#include "structs.h"
#include "mps_reader.h"

namespace py = pybind11;

/**
 * @brief Python wrapper class for HPRLP Parameters
 */
class PyParameter {
public:
    int max_iter;
    double stop_tol;
    double time_limit;
    int device_number;
    int check_iter;
    bool use_Ruiz_scaling;
    bool use_Pock_Chambolle_scaling;
    bool use_bc_scaling;

    PyParameter() 
        : max_iter(INT32_MAX),
          stop_tol(1e-4),
          time_limit(3600.0),
          device_number(0),
          check_iter(150),
          use_Ruiz_scaling(true),
          use_Pock_Chambolle_scaling(true),
          use_bc_scaling(true) {}

    // Convert to C Parameter struct
    HPRLP_parameters to_c_struct() const {
        HPRLP_parameters param;
        param.max_iter = max_iter;
        param.stop_tol = stop_tol;
        param.time_limit = time_limit;
        param.device_number = device_number;
        param.check_iter = check_iter;
        param.use_Ruiz_scaling = use_Ruiz_scaling;
        param.use_Pock_Chambolle_scaling = use_Pock_Chambolle_scaling;
        param.use_bc_scaling = use_bc_scaling;
        return param;
    }
};

/**
 * @brief Python wrapper class for HPRLP Results
 */
class PyResults {
public:
    double residuals;
    double primal_obj;
    double gap;
    double time4;
    double time6;
    double time8;
    double time;
    int iter4;
    int iter6;
    int iter8;
    int iter;
    std::string status;
    std::vector<double> x;  // Primal solution
    std::vector<double> y;  // Dual solution

    PyResults() 
        : residuals(0), primal_obj(0), gap(0),
          time4(0), time6(0), time8(0), time(0),
          iter4(0), iter6(0), iter8(0), iter(0),
          status("UNKNOWN") {}

    // Construct from C HPRLP_results struct
    static PyResults from_c_struct(const HPRLP_results& result, int n, int m) {
        PyResults py_result;
        py_result.residuals = result.residuals;
        py_result.primal_obj = result.primal_obj;
        py_result.gap = result.gap;
        py_result.time4 = result.time4;
        py_result.time6 = result.time6;
        py_result.time8 = result.time8;
        py_result.time = result.time;
        py_result.iter4 = result.iter4;
        py_result.iter6 = result.iter6;
        py_result.iter8 = result.iter8;
        py_result.iter = result.iter;
        py_result.status = std::string(result.status);  // Convert char array to string

        // Copy solution vectors
        if (result.x != nullptr) {
            py_result.x = std::vector<double>(result.x, result.x + n);
        }
        if (result.y != nullptr) {
            py_result.y = std::vector<double>(result.y, result.y + m);
        }

        return py_result;
    }

    py::dict to_dict() const {
        py::dict d;
        d["residuals"] = residuals;
        d["primal_obj"] = primal_obj;
        d["gap"] = gap;
        d["time4"] = time4;
        d["time6"] = time6;
        d["time8"] = time8;
        d["time"] = time;
        d["iter4"] = iter4;
        d["iter6"] = iter6;
        d["iter8"] = iter8;
        d["iter"] = iter;
        d["status"] = status;
        d["x"] = x;
        d["y"] = y;
        return d;
    }
};

/**
 * @brief Python wrapper class for LP Model
 */
class PyModel {
public:
    LP_info_cpu* model_ptr;
    
    PyModel() : model_ptr(nullptr) {}
    
    ~PyModel() {
        // Destructor will be called by Python's garbage collector
        // But we also provide an explicit free() method
    }
    
    bool is_valid() const {
        return model_ptr != nullptr;
    }
    
    int get_m() const {
        return model_ptr ? model_ptr->m : 0;
    }
    
    int get_n() const {
        return model_ptr ? model_ptr->n : 0;
    }
    
    double get_obj_constant() const {
        return model_ptr ? model_ptr->obj_constant : 0.0;
    }
};

/**
 * @brief Create model from numpy arrays (CSR format) - Python wrapper
 */
PyModel py_create_model_from_arrays(
    int m, int n, int nnz,
    py::array_t<int> rowPtr_arr,
    py::array_t<int> colIndex_arr,
    py::array_t<double> values_arr,
    py::array_t<double> AL_arr,
    py::array_t<double> AU_arr,
    py::array_t<double> l_arr,
    py::array_t<double> u_arr,
    py::array_t<double> c_arr,
    bool is_csc = false
) {
    PyModel py_model;
    
    // Validate dimensions
    if (m <= 0 || n <= 0 || nnz <= 0) {
        throw std::invalid_argument("Invalid dimensions: m, n, and nnz must be positive");
    }

    // Get buffer pointers
    auto rowPtr_buf = rowPtr_arr.request();
    auto colIndex_buf = colIndex_arr.request();
    auto values_buf = values_arr.request();
    auto AL_buf = AL_arr.request();
    auto AU_buf = AU_arr.request();
    auto l_buf = l_arr.request();
    auto u_buf = u_arr.request();
    auto c_buf = c_arr.request();

    // Validate array sizes
    int expected_ptr_size = is_csc ? n + 1 : m + 1;
    if (rowPtr_buf.size != expected_ptr_size) {
        throw std::invalid_argument("Invalid rowPtr/colPtr size");
    }
    if (colIndex_buf.size != nnz || values_buf.size != nnz) {
        throw std::invalid_argument("Invalid colIndex/values size");
    }
    if (AL_buf.size != m || AU_buf.size != m) {
        throw std::invalid_argument("Invalid AL/AU size");
    }
    if (l_buf.size != n || u_buf.size != n || c_buf.size != n) {
        throw std::invalid_argument("Invalid l/u/c size");
    }

    // Get raw pointers
    int* rowPtr = static_cast<int*>(rowPtr_buf.ptr);
    int* colIndex = static_cast<int*>(colIndex_buf.ptr);
    double* values = static_cast<double*>(values_buf.ptr);
    double* AL = static_cast<double*>(AL_buf.ptr);
    double* AU = static_cast<double*>(AU_buf.ptr);
    double* l = static_cast<double*>(l_buf.ptr);
    double* u = static_cast<double*>(u_buf.ptr);
    double* c = static_cast<double*>(c_buf.ptr);

    // Create model using C API
    py_model.model_ptr = create_model_from_arrays(
        m, n, nnz,
        rowPtr, colIndex, values,
        AL, AU, l, u, c,
        is_csc
    );
    
    if (py_model.model_ptr == nullptr) {
        throw std::runtime_error("Failed to create model from arrays");
    }

    return py_model;
}

/**
 * @brief Create model from MPS file - Python wrapper
 */
PyModel py_create_model_from_mps(const std::string& filename) {
    PyModel py_model;
    
    // Create model using C API
    py_model.model_ptr = create_model_from_mps(filename.c_str());
    
    if (py_model.model_ptr == nullptr) {
        throw std::runtime_error("Failed to create model from MPS file: " + filename);
    }
    
    return py_model;
}

/**
 * @brief Solve model
 */
PyResults solve_model_py(const PyModel& py_model, const PyParameter* py_param_ptr) {
    if (!py_model.is_valid()) {
        throw std::invalid_argument("Invalid model: model is null");
    }
    
    // Convert parameter to core parameter (or use NULL for defaults)
    HPRLP_parameters core_param;
    HPRLP_parameters* param_ptr = nullptr;
    
    if (py_param_ptr != nullptr) {
        core_param = py_param_ptr->to_c_struct();
        param_ptr = &core_param;
    }
    
    // Call the C++ solver
    HPRLP_results result = solve(py_model.model_ptr, param_ptr);
    
    // Get dimensions from model
    int n = py_model.get_n();
    int m = py_model.get_m();

    // Convert to Python result
    PyResults py_result = PyResults::from_c_struct(result, n, m);

    // Free C result memory (x and y were allocated by solve)
    if (result.x != nullptr) {
        free(result.x);
    }
    if (result.y != nullptr) {
        free(result.y);
    }

    return py_result;
}

/**
 * @brief Free model
 */
void free_model_py(PyModel& py_model) {
    if (py_model.is_valid()) {
        free_model(py_model.model_ptr);
        py_model.model_ptr = nullptr;
    }
}

/**
 * @brief Pybind11 module definition
 */
PYBIND11_MODULE(_hprlp_core, m) {
    m.doc() = "HPRLP: Halpern-Peaceman-Rachford Linear Programming solver with GPU acceleration";

    // Parameters class
    py::class_<PyParameter>(m, "Parameters", "Solver parameters for HPRLP")
        .def(py::init<>())
        .def_readwrite("max_iter", &PyParameter::max_iter, 
                      "Maximum number of iterations (default: INT_MAX)")
        .def_readwrite("stop_tol", &PyParameter::stop_tol,
                      "Stopping tolerance (default: 1e-4)")
        .def_readwrite("time_limit", &PyParameter::time_limit,
                      "Time limit in seconds (default: 3600.0)")
        .def_readwrite("device_number", &PyParameter::device_number,
                      "CUDA device number (default: 0)")
        .def_readwrite("check_iter", &PyParameter::check_iter,
                      "Iterations between convergence checks (default: 150)")
        .def_readwrite("use_Ruiz_scaling", &PyParameter::use_Ruiz_scaling,
                      "Use Ruiz scaling (default: True)")
        .def_readwrite("use_Pock_Chambolle_scaling", &PyParameter::use_Pock_Chambolle_scaling,
                      "Use Pock-Chambolle scaling (default: True)")
        .def_readwrite("use_bc_scaling", &PyParameter::use_bc_scaling,
                      "Use bound constraint scaling (default: True)")
        .def("__repr__", [](const PyParameter &p) {
            return "<HPRLP.Parameters max_iter=" + std::to_string(p.max_iter) +
                   " stop_tol=" + std::to_string(p.stop_tol) + ">";
        });

    // Results class
    py::class_<PyResults>(m, "Results", "Results from HPRLP solver")
        .def(py::init<>())
        .def_readonly("residuals", &PyResults::residuals, "Final residuals")
        .def_readonly("primal_obj", &PyResults::primal_obj, "Primal objective value")
        .def_readonly("gap", &PyResults::gap, "Duality gap")
        .def_readonly("time4", &PyResults::time4, "Time to reach 1e-4 tolerance")
        .def_readonly("time6", &PyResults::time6, "Time to reach 1e-6 tolerance")
        .def_readonly("time8", &PyResults::time8, "Time to reach 1e-8 tolerance")
        .def_readonly("time", &PyResults::time, "Total solve time")
        .def_readonly("iter4", &PyResults::iter4, "Iterations to reach 1e-4")
        .def_readonly("iter6", &PyResults::iter6, "Iterations to reach 1e-6")
        .def_readonly("iter8", &PyResults::iter8, "Iterations to reach 1e-8")
        .def_readonly("iter", &PyResults::iter, "Total iterations")
        .def_readonly("status", &PyResults::status, "Solver status")
        .def_readonly("x", &PyResults::x, "Primal solution vector")
        .def_readonly("y", &PyResults::y, "Dual solution vector")
        .def("to_dict", &PyResults::to_dict, "Convert results to dictionary")
        .def("__repr__", [](const PyResults &r) {
            return "<HPRLP.Results status='" + r.status + 
                   "' iter=" + std::to_string(r.iter) +
                   " time=" + std::to_string(r.time) + "s>";
        });

    // Model class
    py::class_<PyModel>(m, "Model", "LP model for HPRLP solver")
        .def(py::init<>())
        .def("is_valid", &PyModel::is_valid, "Check if model is valid")
        .def_property_readonly("m", &PyModel::get_m, "Number of constraints")
        .def_property_readonly("n", &PyModel::get_n, "Number of variables")
        .def_property_readonly("obj_constant", &PyModel::get_obj_constant, "Objective constant term")
        .def("__repr__", [](const PyModel &model) -> std::string {
            if (model.is_valid()) {
                return "<HPRLP.Model m=" + std::to_string(model.get_m()) +
                       " n=" + std::to_string(model.get_n()) + ">";
            } else {
                return std::string("<HPRLP.Model (invalid)>");
            }
        });

    // Model construction functions
    m.def("create_model_from_arrays", &py_create_model_from_arrays,
          py::arg("m"),
          py::arg("n"),
          py::arg("nnz"),
          py::arg("rowPtr"),
          py::arg("colIndex"),
          py::arg("values"),
          py::arg("AL"),
          py::arg("AU"),
          py::arg("l"),
          py::arg("u"),
          py::arg("c"),
          py::arg("is_csc") = false,
          R"pbdoc(
        Create an LP model from numpy arrays.

        The LP problem has the form:
            minimize    c'*x
            subject to  AL <= A*x <= AU
                        l <= x <= u

        Parameters
        ----------
        m : int
            Number of constraints
        n : int
            Number of variables
        nnz : int
            Number of non-zero elements in constraint matrix
        rowPtr : np.ndarray[int]
            Row pointer array (CSR) or column pointer (CSC)
        colIndex : np.ndarray[int]
            Column indices (CSR) or row indices (CSC)
        values : np.ndarray[float]
            Non-zero values
        AL : np.ndarray[float]
            Lower bounds for constraints (use -np.inf for unbounded)
        AU : np.ndarray[float]
            Upper bounds for constraints (use np.inf for unbounded)
        l : np.ndarray[float]
            Lower bounds for variables (use -np.inf for unbounded)
        u : np.ndarray[float]
            Upper bounds for variables (use np.inf for unbounded)
        c : np.ndarray[float]
            Objective coefficients
        is_csc : bool, optional
            Whether matrix is in CSC format (default: False, CSR)

        Returns
        -------
        Model
            LP model object
        )pbdoc");

    m.def("create_model_from_mps", &py_create_model_from_mps,
          py::arg("filename"),
          R"pbdoc(
        Create an LP model from an MPS file.

        Parameters
        ----------
        filename : str
            Path to the MPS file

        Returns
        -------
        Model
            LP model object
        )pbdoc");

    m.def("solve", &solve_model_py,
          py::arg("model"),
          py::arg("param") = nullptr,
          R"pbdoc(
        Solve an LP model.

        Parameters
        ----------
        model : Model
            LP model to solve
        param : Parameters, optional
            Solver parameters (default: None, uses default parameters)

        Returns
        -------
        Results
            Solver results including solution vectors and statistics
        )pbdoc");

    m.def("free_model", &free_model_py,
          py::arg("model"),
          R"pbdoc(
        Free an LP model and release memory.

        Parameters
        ----------
        model : Model
            LP model to free
        )pbdoc");

    // Constants
    m.attr("__version__") = "0.1.1";
    m.attr("INFINITY") = std::numeric_limits<double>::infinity();
}
