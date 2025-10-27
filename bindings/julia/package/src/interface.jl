"""
High-level Julia interface for HPRLP.

This module provides user-friendly Julia types and functions.
"""

"""
    Model

Represents an LP problem model.

# Fields
- `ptr::Ptr{Cvoid}`: Pointer to the C LP_info_cpu struct
- `m::Int`: Number of constraints
- `n::Int`: Number of variables
- `obj_constant::Float64`: Constant term in objective

# Methods
- `Model(A, AL, AU, l, u, c)`: Create model from arrays
- `Model(filename)`: Create model from MPS file
- `solve(model, params)`: Solve the model
- `free(model)`: Free model memory (called automatically by finalizer)

# Example
```julia
using HPRLP
using SparseArrays

# Create model from arrays
A = sparse([1.0 2.0; 3.0 1.0])
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]
c = [-3.0, -5.0]

model = Model(A, AL, AU, l, u, c)
result = solve(model)
println("Optimal value: ", result.primal_obj)

# Model is automatically freed when garbage collected
# Or explicitly: free(model)
```
"""
mutable struct Model
    ptr::Ptr{Cvoid}
    m::Int
    n::Int
    obj_constant::Float64
    
    function Model(ptr::Ptr{Cvoid}, m::Int, n::Int, obj_constant::Float64 = 0.0)
        model = new(ptr, m, n, obj_constant)
        # Register finalizer to automatically free memory
        finalizer(free, model)
        return model
    end
end

"""
    Model(A, AL, AU, l, u, c; obj_constant=0.0)

Create an LP model from arrays.

Represents the LP:
```
minimize    c'x + obj_constant
subject to  AL <= Ax <= AU
            l <= x <= u
```

# Arguments
- `A`: Constraint matrix (sparse or dense, m×n)
- `AL`: Lower bounds on constraints (length m, use -Inf for unbounded)
- `AU`: Upper bounds on constraints (length m, use Inf for unbounded)
- `l`: Lower bounds on variables (length n, use -Inf for unbounded)
- `u`: Upper bounds on variables (length n, use Inf for unbounded)
- `c`: Objective coefficients (length n)
- `obj_constant`: Constant term in objective (default: 0.0)

# Returns
- `Model` object
"""
function Model(A::AbstractMatrix{Float64},
               AL::AbstractVector{Float64},
               AU::AbstractVector{Float64},
               l::AbstractVector{Float64},
               u::AbstractVector{Float64},
               c::AbstractVector{Float64};
               obj_constant::Float64 = 0.0)
    
    # Get dimensions
    m, n = size(A)
    
    # Validate dimensions
    @assert length(AL) == m "AL must have length m"
    @assert length(AU) == m "AU must have length m"
    @assert length(l) == n "l must have length n"
    @assert length(u) == n "u must have length n"
    @assert length(c) == n "c must have length n"
    
    # Convert to CSR format
    # Julia uses CSC natively, so we transpose to get CSR
    A_sparse = issparse(A) ? A : sparse(A)
    A_csc = SparseMatrixCSC(A_sparse)  # Ensure it's CSC
    
    # Transpose to convert CSC to CSR (transpose of CSC is effectively CSR)
    A_csr_t = sparse(A_csc')  # Materialize the transpose as sparse
    
    # Now A_csr_t is a CSC matrix which represents the transpose
    # So A_csr_t.colptr is actually the rowPtr for CSR format of original A
    rowPtr = convert(Vector{Int32}, A_csr_t.colptr .- 1)  # Convert to 0-based indexing
    colIndex = convert(Vector{Int32}, A_csr_t.rowval .- 1)  # Convert to 0-based indexing
    values = A_csr_t.nzval
    nnz = length(values)
    
    # Call C function to create model
    ptr = c_create_model_from_arrays(m, n, nnz,
                                      rowPtr, colIndex, values,
                                      AL, AU, l, u, c,
                                      false)  # false = CSR format
    
    if ptr == C_NULL
        error("Failed to create model from arrays")
    end
    
    return Model(ptr, m, n, obj_constant)
end

"""
    Model(filename::String)

Create an LP model from an MPS file.

# Arguments
- `filename`: Path to the MPS file

# Returns
- `Model` object
"""
function Model(filename::String)
    # Check file exists
    if !isfile(filename)
        error("MPS file not found: $filename")
    end
    
    # Call C function to create model
    ptr = c_create_model_from_mps(filename)
    
    if ptr == C_NULL
        error("Failed to create model from MPS file: $filename")
    end
    
    # Parse MPS file to get dimensions
    m, n = get_mps_dimensions(filename)
    
    return Model(ptr, m, n, 0.0)
end

"""
    solve(model::Model, params::Parameters = Parameters())

Solve the LP model.

# Arguments
- `model`: Model object to solve
- `params`: Optional Parameters object

# Returns
- `Results` object containing solution and statistics
"""
function solve(model::Model, params = nothing)
    if model.ptr == C_NULL
        error("Cannot solve freed model")
    end
    
    # Convert parameters to C struct if provided
    c_params = params === nothing ? nothing : to_c_struct(params)
    
    # Call C function to solve
    c_results = c_solve_model(model.ptr, c_params)
    
    # Convert results to Julia struct
    result = from_c_struct(c_results, model.n, model.m)
    
    # Adjust objective value by constant
    result = Results(
        result.x, result.y, result.status,
        result.primal_obj + model.obj_constant,
        result.gap, result.residuals, result.iter, result.time,
        result.iter4, result.iter6, result.iter8,
        result.time4, result.time6, result.time8
    )
    
    return result
end

"""
    free(model::Model)

Free the model's memory.

Note: This is called automatically by the finalizer when the model is garbage collected.
You typically don't need to call this explicitly unless you want to free memory immediately.
"""
function free(model::Model)
    if model.ptr != C_NULL
        c_free_model(model.ptr)
        model.ptr = C_NULL
    end
end

"""
    Parameters

Solver configuration parameters.

# Fields
- `max_iter::Int`: Maximum number of iterations (default: typemax(Int32))
- `stop_tol::Float64`: Stopping tolerance (default: 1e-4)
- `time_limit::Float64`: Time limit in seconds (default: 3600.0)
- `device_number::Int`: CUDA device ID (default: 0)
- `check_iter::Int`: Iterations between convergence checks (default: 150)
- `use_Ruiz_scaling::Bool`: Enable Ruiz equilibration scaling (default: true)
- `use_Pock_Chambolle_scaling::Bool`: Enable Pock-Chambolle scaling (default: true)
- `use_bc_scaling::Bool`: Enable bounds/cost scaling (default: true)

# Example
```julia
params = Parameters(
    max_iter = 10000,
    stop_tol = 1e-9,
    time_limit = 7200.0,
    device_number = 1,
    use_Ruiz_scaling = false
)
```
"""
mutable struct Parameters
    max_iter::Int
    stop_tol::Float64
    time_limit::Float64
    device_number::Int
    check_iter::Int
    use_Ruiz_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    use_bc_scaling::Bool
    
    function Parameters(;
        max_iter::Int = Int(typemax(Int32)),
        stop_tol::Float64 = 1e-4,
        time_limit::Float64 = 3600.0,
        device_number::Int = 0,
        check_iter::Int = 150,
        use_Ruiz_scaling::Bool = true,
        use_Pock_Chambolle_scaling::Bool = true,
        use_bc_scaling::Bool = true)
        
        new(Int(max_iter), stop_tol, time_limit, Int(device_number), Int(check_iter),
            use_Ruiz_scaling, use_Pock_Chambolle_scaling, use_bc_scaling)
    end
end

"""
Convert Julia Parameters to C struct
"""
function to_c_struct(params::Parameters)
    c_params = C_HPRLP_parameters()
    c_params.max_iter = Int32(params.max_iter)
    c_params.stop_tol = params.stop_tol
    c_params.time_limit = params.time_limit
    c_params.device_number = Int32(params.device_number)
    c_params.check_iter = Int32(params.check_iter)
    c_params.use_Ruiz_scaling = params.use_Ruiz_scaling
    c_params.use_Pock_Chambolle_scaling = params.use_Pock_Chambolle_scaling
    c_params.use_bc_scaling = params.use_bc_scaling
    
    return c_params
end

"""
    Results

Solution results from the solver.

# Fields
- `x::Vector{Float64}`: Primal solution
- `y::Vector{Float64}`: Dual solution
- `status::String`: Solution status ("OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", etc.)
- `primal_obj::Float64`: Primal objective value
- `gap::Float64`: Duality gap
- `residuals::Float64`: Residuals
- `iter::Int`: Total iterations
- `time::Float64`: Total solution time (seconds)
- `iter4::Int`: Iterations to 1e-4 tolerance
- `iter6::Int`: Iterations to 1e-6 tolerance
- `iter8::Int`: Iterations to 1e-8 tolerance
- `time4::Float64`: Time to 1e-4 tolerance
- `time6::Float64`: Time to 1e-6 tolerance
- `time8::Float64`: Time to 1e-8 tolerance

# Methods
- `is_optimal(result)`: Check if solution is optimal
"""
struct Results
    x::Vector{Float64}
    y::Vector{Float64}
    status::String
    primal_obj::Float64
    gap::Float64
    residuals::Float64
    iter::Int
    time::Float64
    iter4::Int
    iter6::Int
    iter8::Int
    time4::Float64
    time6::Float64
    time8::Float64
end

"""
Convert C results to Julia Results struct
"""
function from_c_struct(c_results::C_HPRLP_results, n::Int, m::Int)
    # Copy solution vectors
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, m)
    
    if c_results.x != C_NULL
        unsafe_copyto!(pointer(x), c_results.x, n)
    end
    
    if c_results.y != C_NULL
        unsafe_copyto!(pointer(y), c_results.y, m)
    end
    
    # Get status string from char array
    # Convert NTuple{64, UInt8} to String, stopping at null terminator
    status_bytes = collect(c_results.status)
    null_idx = findfirst(==(0x00), status_bytes)
    if null_idx !== nothing
        status_bytes = status_bytes[1:null_idx-1]
    end
    status = String(status_bytes)
    
    # Create Results object
    result = Results(
        x, y, status,
        c_results.primal_obj,
        c_results.gap,
        c_results.residuals,
        c_results.iter,
        c_results.time,
        c_results.iter4,
        c_results.iter6,
        c_results.iter8,
        c_results.time4,
        c_results.time6,
        c_results.time8
    )
    
    # Free C memory
    c_free_results(c_results.x, c_results.y)
    
    return result
end