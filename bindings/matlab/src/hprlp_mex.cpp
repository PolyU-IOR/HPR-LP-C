/**
 * @file hprlp_mex.cpp
 * @brief MATLAB MEX interface for HPRLP solver
 * 
 * This MEX file provides the low-level interface between MATLAB and the
 * HPRLP C++/CUDA solver library.
 */

#include "mex.h"
#include "matrix.h"
#include "HPRLP.h"
#include "mps_reader.h"
#include "preprocess.h"
#include <cstring>
#include <string>
#include <cmath>
#include <sstream>
#include <iostream>

/**
 * Custom streambuf that redirects to mexPrintf
 */
class MexStreambuf : public std::streambuf {
protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        mexPrintf("%.*s", static_cast<int>(n), s);
        mexEvalString("drawnow;");  // Force MATLAB to update display
        return n;
    }
    
    virtual int overflow(int c) override {
        if (c != EOF) {
            char ch = static_cast<char>(c);
            mexPrintf("%c", ch);
            if (c == '\n') {
                mexEvalString("drawnow;");
            }
        }
        return c;
    }
};

/**
 * Helper function to extract a scalar double from mxArray
 */
double getScalarDouble(const mxArray* arr, const char* name) {
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxGetNumberOfElements(arr) != 1) {
        mexErrMsgIdAndTxt("HPRLP:InvalidInput", 
                         "Parameter '%s' must be a real scalar double", name);
    }
    return mxGetScalar(arr);
}

/**
 * Helper function to extract an integer from mxArray
 */
int getScalarInt(const mxArray* arr, const char* name) {
    double val = getScalarDouble(arr, name);
    return static_cast<int>(val);
}

/**
 * Helper function to extract a boolean from mxArray
 */
bool getScalarBool(const mxArray* arr, const char* name) {
    if (!mxIsLogicalScalar(arr) && !mxIsDouble(arr)) {
        mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                         "Parameter '%s' must be a logical or numeric scalar", name);
    }
    if (mxIsLogicalScalar(arr)) {
        return mxIsLogicalScalarTrue(arr);
    }
    return getScalarDouble(arr, name) != 0.0;
}

/**
 * Helper function to extract a string from mxArray
 */
std::string getString(const mxArray* arr, const char* name) {
    if (!mxIsChar(arr)) {
        mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                         "Parameter '%s' must be a string", name);
    }
    char* str = mxArrayToString(arr);
    if (str == NULL) {
        mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                         "Failed to convert '%s' to string", name);
    }
    std::string result(str);
    mxFree(str);
    return result;
}

/**
 * MEX gateway function
 * 
 * Available commands:
 *   - create_model_from_arrays: Create LP model from matrices/vectors
 *   - create_model_from_mps: Create LP model from MPS file
 *   - solve: Solve the LP model
 *   - get_model_info: Get model dimensions and info
 *   - free_model: Free model memory
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    
    // Redirect std::cout to MATLAB console for this MEX call
    static std::streambuf* cout_original = nullptr;
    static MexStreambuf mexbuf;
    
    // First time setup: redirect cout
    if (cout_original == nullptr) {
        cout_original = std::cout.rdbuf(&mexbuf);
    }
    
    // Check minimum number of arguments
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                         "First argument must be a command string");
    }
    
    // Get command
    std::string command = getString(prhs[0], "command");
    
    // ========================================================================
    // Model-based API Commands
    // ========================================================================
    
    // Available commands:
    //   - create_model_from_arrays: Create LP model from matrices/vectors
    //   - create_model_from_mps: Create LP model from MPS file
    //   - solve: Solve the LP model
    //   - get_model_info: Get model dimensions and info
    //   - free_model: Free model memory
    // ========================================================================
    
    // ========================================================================
    // Command: create_model_from_arrays
    // ========================================================================
    if (command == "create_model_from_arrays") {
        if (nrhs != 7) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                             "create_model_from_arrays requires 7 arguments: command, A, AL, AU, l, u, c");
        }
        
        // Get input matrices and vectors
        const mxArray* A_mx = prhs[1];
        const mxArray* AL_mx = prhs[2];
        const mxArray* AU_mx = prhs[3];
        const mxArray* l_mx = prhs[4];
        const mxArray* u_mx = prhs[5];
        const mxArray* c_mx = prhs[6];
        
        // Validate A is sparse
        if (!mxIsSparse(A_mx)) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Matrix A must be sparse");
        }
        
        // Get dimensions
        int m = static_cast<int>(mxGetM(A_mx));
        int n = static_cast<int>(mxGetN(A_mx));
        int nnz = static_cast<int>(mxGetNzmax(A_mx));
        
        // Validate dimensions
        if (mxGetM(AL_mx) * mxGetN(AL_mx) != m) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "AL must have length m");
        }
        if (mxGetM(AU_mx) * mxGetN(AU_mx) != m) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "AU must have length m");
        }
        if (mxGetM(l_mx) * mxGetN(l_mx) != n) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "l must have length n");
        }
        if (mxGetM(u_mx) * mxGetN(u_mx) != n) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "u must have length n");
        }
        if (mxGetM(c_mx) * mxGetN(c_mx) != n) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "c must have length n");
        }
        
        // Extract sparse matrix in CSC format (MATLAB's native format)
        mwIndex* jc = mxGetJc(A_mx);  // Column pointers
        mwIndex* ir = mxGetIr(A_mx);  // Row indices
        double* pr = mxGetPr(A_mx);   // Values
        
        // Allocate arrays for CSC format
        int* colPtr = new int[n + 1];
        int* rowIndex = new int[nnz];
        double* values = new double[nnz];
        
        // Convert to 0-based indexing (copy the data)
        for (int j = 0; j <= n; j++) {
            colPtr[j] = static_cast<int>(jc[j]);
        }
        for (int i = 0; i < nnz; i++) {
            rowIndex[i] = static_cast<int>(ir[i]);
            values[i] = pr[i];
        }
        
        // Get vectors
        double* AL = mxGetPr(AL_mx);
        double* AU = mxGetPr(AU_mx);
        double* l = mxGetPr(l_mx);
        double* u = mxGetPr(u_mx);
        double* c = mxGetPr(c_mx);
        
        // Call C library function (use CSC format, is_csc = true)
        LP_info_cpu* model = create_model_from_arrays(
            m, n, nnz,
            colPtr, rowIndex, values,
            AL, AU, l, u, c,
            true  // is_csc = true for MATLAB's CSC format
        );
        
        // Clean up temporary arrays
        delete[] colPtr;
        delete[] rowIndex;
        delete[] values;
        
        if (model == NULL) {
            mexErrMsgIdAndTxt("HPRLP:RuntimeError", "Failed to create model from arrays");
        }
        
        // Return model pointer as uint64
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        uint64_t* ptr = static_cast<uint64_t*>(mxGetData(plhs[0]));
        *ptr = reinterpret_cast<uint64_t>(model);
        
        return;
    }
    
    // ========================================================================
    // Command: create_model_from_mps
    // ========================================================================
    if (command == "create_model_from_mps") {
        if (nrhs != 2) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                             "create_model_from_mps requires 2 arguments: command, filename");
        }
        
        // Get filename
        std::string filename = getString(prhs[1], "filename");
        
        // Call C library function
        LP_info_cpu* model = create_model_from_mps(filename.c_str());
        
        if (model == NULL) {
            mexErrMsgIdAndTxt("HPRLP:RuntimeError", 
                             "Failed to create model from MPS file: %s", filename.c_str());
        }
        
        // Return model pointer as uint64
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        uint64_t* ptr = static_cast<uint64_t*>(mxGetData(plhs[0]));
        *ptr = reinterpret_cast<uint64_t>(model);
        
        return;
    }
    
    // ========================================================================
    // Command: solve
    // ========================================================================
    if (command == "solve") {
        if (nrhs < 2 || nrhs > 3) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                             "solve requires 2-3 arguments: command, model_handle, [param_struct]");
        }
        
        // Get model pointer from uint64 handle
        if (!mxIsUint64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Model handle must be a uint64 scalar");
        }
        uint64_t* ptr = static_cast<uint64_t*>(mxGetData(prhs[1]));
        LP_info_cpu* model = reinterpret_cast<LP_info_cpu*>(*ptr);
        
        if (model == NULL) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Invalid model handle (NULL pointer)");
        }
        
        // Get parameters (or use defaults)
        HPRLP_parameters* param = nullptr;
        HPRLP_parameters param_storage;
        
        if (nrhs >= 3 && !mxIsEmpty(prhs[2])) {
            param = &param_storage;
            const mxArray* param_struct = prhs[2];
            
            // Extract parameter fields
            mxArray* field;
            if ((field = mxGetField(param_struct, 0, "max_iter")) != NULL) {
                param->max_iter = getScalarInt(field, "max_iter");
            }
            if ((field = mxGetField(param_struct, 0, "stop_tol")) != NULL) {
                param->stop_tol = getScalarDouble(field, "stop_tol");
            }
            if ((field = mxGetField(param_struct, 0, "time_limit")) != NULL) {
                param->time_limit = getScalarDouble(field, "time_limit");
            }
            if ((field = mxGetField(param_struct, 0, "device_number")) != NULL) {
                param->device_number = getScalarInt(field, "device_number");
            }
            if ((field = mxGetField(param_struct, 0, "check_iter")) != NULL) {
                param->check_iter = getScalarInt(field, "check_iter");
            }
            if ((field = mxGetField(param_struct, 0, "use_Ruiz_scaling")) != NULL) {
                param->use_Ruiz_scaling = getScalarBool(field, "use_Ruiz_scaling");
            }
            if ((field = mxGetField(param_struct, 0, "use_Pock_Chambolle_scaling")) != NULL) {
                param->use_Pock_Chambolle_scaling = getScalarBool(field, "use_Pock_Chambolle_scaling");
            }
            if ((field = mxGetField(param_struct, 0, "use_bc_scaling")) != NULL) {
                param->use_bc_scaling = getScalarBool(field, "use_bc_scaling");
            }
        }
        
        // Call C library function
        HPRLP_results result = solve(model, param);
        
        // Create output structure
        const char* field_names[] = {"status", "residuals", "primal_obj", "gap",
                                    "time4", "time6", "time8", "time",
                                    "iter4", "iter6", "iter8", "iter",
                                    "x", "y"};
        plhs[0] = mxCreateStructMatrix(1, 1, 14, field_names);
        
        // Set status
        mxSetField(plhs[0], 0, "status", mxCreateString(result.status));
        
        // Set scalar values
        mxSetField(plhs[0], 0, "residuals", mxCreateDoubleScalar(result.residuals));
        mxSetField(plhs[0], 0, "primal_obj", mxCreateDoubleScalar(result.primal_obj));
        mxSetField(plhs[0], 0, "gap", mxCreateDoubleScalar(result.gap));
        mxSetField(plhs[0], 0, "time4", mxCreateDoubleScalar(result.time4));
        mxSetField(plhs[0], 0, "time6", mxCreateDoubleScalar(result.time6));
        mxSetField(plhs[0], 0, "time8", mxCreateDoubleScalar(result.time8));
        mxSetField(plhs[0], 0, "time", mxCreateDoubleScalar(result.time));
        mxSetField(plhs[0], 0, "iter4", mxCreateDoubleScalar(result.iter4));
        mxSetField(plhs[0], 0, "iter6", mxCreateDoubleScalar(result.iter6));
        mxSetField(plhs[0], 0, "iter8", mxCreateDoubleScalar(result.iter8));
        mxSetField(plhs[0], 0, "iter", mxCreateDoubleScalar(result.iter));
        
        // Get dimensions from model
        int m = model->m;
        int n = model->n;
        
        // Set primal solution x
        if (result.x != NULL) {
            mxArray* x_array = mxCreateDoubleMatrix(n, 1, mxREAL);
            double* x_ptr = mxGetPr(x_array);
            for (int i = 0; i < n; i++) {
                x_ptr[i] = result.x[i];
            }
            mxSetField(plhs[0], 0, "x", x_array);
            delete[] result.x;  // Free C++ allocated memory
        } else {
            mxSetField(plhs[0], 0, "x", mxCreateDoubleMatrix(0, 0, mxREAL));
        }
        
        // Set dual solution y
        if (result.y != NULL) {
            mxArray* y_array = mxCreateDoubleMatrix(m, 1, mxREAL);
            double* y_ptr = mxGetPr(y_array);
            for (int i = 0; i < m; i++) {
                y_ptr[i] = result.y[i];
            }
            mxSetField(plhs[0], 0, "y", y_array);
            delete[] result.y;  // Free C++ allocated memory
        } else {
            mxSetField(plhs[0], 0, "y", mxCreateDoubleMatrix(0, 0, mxREAL));
        }
        
        return;
    }
    
    // ========================================================================
    // Command: get_model_info
    // ========================================================================
    if (command == "get_model_info") {
        if (nrhs != 2) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                             "get_model_info requires 2 arguments: command, model_handle");
        }
        
        // Get model pointer from uint64 handle
        if (!mxIsUint64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Model handle must be a uint64 scalar");
        }
        uint64_t* ptr = static_cast<uint64_t*>(mxGetData(prhs[1]));
        LP_info_cpu* model = reinterpret_cast<LP_info_cpu*>(*ptr);
        
        if (model == NULL) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Invalid model handle (NULL pointer)");
        }
        
        // Create output structure
        const char* field_names[] = {"m", "n", "obj_constant"};
        plhs[0] = mxCreateStructMatrix(1, 1, 3, field_names);
        
        // Set model information
        mxSetField(plhs[0], 0, "m", mxCreateDoubleScalar(model->m));
        mxSetField(plhs[0], 0, "n", mxCreateDoubleScalar(model->n));
        mxSetField(plhs[0], 0, "obj_constant", mxCreateDoubleScalar(model->obj_constant));
        
        return;
    }
    
    // ========================================================================
    // Command: free_model
    // ========================================================================
    if (command == "free_model") {
        if (nrhs != 2) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                             "free_model requires 2 arguments: command, model_handle");
        }
        
        // Get model pointer from uint64 handle
        if (!mxIsUint64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1) {
            mexErrMsgIdAndTxt("HPRLP:InvalidInput", "Model handle must be a uint64 scalar");
        }
        uint64_t* ptr = static_cast<uint64_t*>(mxGetData(prhs[1]));
        LP_info_cpu* model = reinterpret_cast<LP_info_cpu*>(*ptr);
        
        if (model != NULL) {
            free_model(model);
            *ptr = 0;  // Set handle to 0 to prevent double-free
        }
        
        return;
    }
    
    // Unknown command
    mexErrMsgIdAndTxt("HPRLP:InvalidInput",
                     "Unknown command '%s'. Use 'create_model_from_arrays', 'create_model_from_mps', 'solve', 'get_model_info', or 'free_model'", command.c_str());
}
