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
    read_time::Float64
    
    function Model(ptr::Ptr{Cvoid}, m::Int, n::Int, obj_constant::Float64 = 0.0,
                   read_time::Float64 = 0.0)
        model = new(ptr, m, n, obj_constant, read_time)
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
        error("LP file not found: $filename")
    end

    lower = lowercase(filename)
    if endswith(lower, ".h5") || endswith(lower, ".hdf5")
        return model_from_hdf5(filename)
    end

    # Call C function to create model
    ptr, read_time = c_create_model_from_mps(filename)
    
    if ptr == C_NULL
        error("Failed to create model from MPS file: $filename")
    end
    
    # Parse MPS file to get dimensions
    m, n = get_mps_dimensions(filename)
    
    return Model(ptr, m, n, 0.0, read_time)
end

function model_from_hdf5(filename::String)
    read_start = time()
    data = h5open(filename, "r") do file
        version = Int32(read(file, "schema_version"))
        version == 1 ||
            error("Unsupported LP HDF5 schema version $version in $filename")
        matrix_size = Vector{Int64}(read(file, "A/size"))
        length(matrix_size) == 2 || error("Invalid A/size dataset in $filename")
        m, n = Int.(matrix_size)
        colptr = Vector{Int32}(read(file, "A/colptr"))
        rowval = Vector{Int32}(read(file, "A/rowval"))
        nzval = Vector{Float64}(read(file, "A/nzval"))
        c = Vector{Float64}(read(file, "c"))
        AL = Vector{Float64}(read(file, "AL"))
        AU = Vector{Float64}(read(file, "AU"))
        l = Vector{Float64}(read(file, "l"))
        u = Vector{Float64}(read(file, "u"))
        obj_constant = Float64(read(file, "obj_constant"))
        return m, n, colptr, rowval, nzval, c, AL, AU, l, u, obj_constant
    end
    m, n, colptr, rowval, nzval, c, AL, AU, l, u, obj_constant = data
    length(colptr) == n + 1 || error("Invalid A/colptr length in $filename")
    length(rowval) == length(nzval) || error("Invalid sparse arrays in $filename")
    length(c) == n || error("Invalid c length in $filename")
    length(l) == n || error("Invalid l length in $filename")
    length(u) == n || error("Invalid u length in $filename")
    length(AL) == m || error("Invalid AL length in $filename")
    length(AU) == m || error("Invalid AU length in $filename")
    first(colptr) == 1 || error("A/colptr must use Julia 1-based indexing")
    last(colptr) == length(nzval) + 1 || error("Invalid final A/colptr")
    all(index -> 1 <= index <= m, rowval) ||
        error("A/rowval contains an out-of-range row index")
    colptr .-= 1
    rowval .-= 1
    ptr = c_create_model_from_arrays_with_obj_constant(
        m, n, length(nzval), colptr, rowval, nzval,
        AL, AU, l, u, c, obj_constant, true,
    )
    ptr == C_NULL && error("Failed to create model from HDF5 file: $filename")
    return Model(ptr, m, n, 0.0, time() - read_start)
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
        result.x, result.y, result.z, result.status,
        result.primal_obj + model.obj_constant,
        result.gap, result.residuals, result.iter, result.time,
        result.iter4, result.iter6, result.iter8,
        result.time4, result.time6, result.time8, result.timing
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
- `CUSPARSE_spmv::Bool`: Force the cuSPARSE SpMVOp path and disable fused-kernel autotuning (default: false)
- `autotune_verbose::Bool`: Print backend autotuning diagnostics when fused kernels are enabled (default: false)
- `use_CR_scaling::Bool`: Enable Curtis-Reid prescaling before the other scaling passes (default: false)
- `use_Ruiz_scaling::Bool`: Enable Ruiz equilibration scaling (default: true)
- `use_Pock_Chambolle_scaling::Bool`: Enable Pock-Chambolle scaling (default: true)
- `use_bc_scaling::Bool`: Enable bounds/cost scaling (default: true)
- `use_presolve::Bool`: Enable embedded PSLP presolve/postsolve (default: true)

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
    CUSPARSE_spmv::Bool
    autotune_verbose::Bool
    enable_progress_monitor::Bool
    enable_progress_control::Bool
    enable_sigma_rebalance_restart::Bool
    use_progress_restart_guard::Bool
    restart_cooldown_checks::Int
    debug_restart::Bool
    debug_sigma::Bool
    fixed_sigma::Float64
    use_CR_scaling::Bool
    use_Ruiz_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    use_bc_scaling::Bool
    use_presolve::Bool
    enable_gpu_folding::Bool
    presolver::Symbol
    use_reduced_matrix::Bool
    auto_reduced_compression_policy::Bool
    print_debug_info::Bool
    specified_parameter_mask::UInt64

    function Parameters(;
        max_iter::Int = Int(typemax(Int32)),
        stop_tol::Float64 = 1e-4,
        time_limit::Float64 = 3600.0,
        device_number::Int = 0,
        check_iter::Int = 150,
        CUSPARSE_spmv::Bool = false,
        autotune_verbose::Bool = false,
        enable_progress_monitor::Bool = true,
        enable_progress_control::Bool = true,
        enable_sigma_rebalance_restart::Bool = true,
        use_progress_restart_guard::Bool = false,
        restart_cooldown_checks::Int = 0,
        debug_restart::Bool = false,
        debug_sigma::Bool = false,
        fixed_sigma::Float64 = NaN,
        use_CR_scaling::Bool = true,
        use_Ruiz_scaling::Bool = true,
        use_Pock_Chambolle_scaling::Bool = true,
        use_bc_scaling::Bool = true,
        use_presolve::Bool = true,
        enable_gpu_folding::Bool = true,
        presolver::Symbol = :gpu,
        use_reduced_matrix::Bool = false,
        auto_reduced_compression_policy::Bool = true,
        print_debug_info::Bool = false,
        specified_parameter_mask::UInt64 = UInt64(0))

        presolver in (:pslp, :gpu, :none) ||
            throw(ArgumentError("presolver must be :pslp, :gpu, or :none"))
        new(Int(max_iter), stop_tol, time_limit, Int(device_number), Int(check_iter),
            CUSPARSE_spmv, autotune_verbose,
            enable_progress_monitor, enable_progress_control,
            enable_sigma_rebalance_restart, use_progress_restart_guard,
            Int(restart_cooldown_checks), debug_restart, debug_sigma, fixed_sigma,
            use_CR_scaling, use_Ruiz_scaling, use_Pock_Chambolle_scaling,
            use_bc_scaling, use_presolve, enable_gpu_folding, presolver,
            use_reduced_matrix, auto_reduced_compression_policy,
            print_debug_info, specified_parameter_mask)
    end
end

"""
Convert Julia Parameters to C struct
"""
function to_c_struct(params::Parameters)
    presolver_value = params.presolver === :pslp ? Int32(0) :
                      params.presolver === :gpu ? Int32(1) : Int32(2)
    use_presolve = params.use_presolve && params.presolver !== :none
    return C_HPRLP_parameters(
        Int32(params.max_iter), params.stop_tol, params.time_limit,
        Int32(params.device_number), Int32(params.check_iter),
        params.CUSPARSE_spmv, params.autotune_verbose,
        params.enable_progress_monitor, params.enable_progress_control,
        params.enable_sigma_rebalance_restart,
        params.use_progress_restart_guard,
        Int32(params.restart_cooldown_checks),
        params.debug_restart, params.debug_sigma, params.fixed_sigma,
        params.use_CR_scaling, params.use_Ruiz_scaling,
        params.use_Pock_Chambolle_scaling, params.use_bc_scaling,
        use_presolve, params.enable_gpu_folding, presolver_value,
        params.use_reduced_matrix,
        params.auto_reduced_compression_policy,
        params.print_debug_info,
        params.specified_parameter_mask)
end

"""
    Results

Solution results from the solver.

# Fields
- `x::Vector{Float64}`: Primal solution
- `y::Vector{Float64}`: Dual solution
- `z::Vector{Float64}`: Bound-dual solution
- `status::String`: Solution status ("OPTIMAL", "TIME_LIMIT", "ITER_LIMIT", etc.)
- `primal_obj::Float64`: Primal objective value
- `gap::Float64`: Duality gap
- `residuals::Float64`: Residuals
- `iter::Int`: Total iterations
- `time::Float64`: Main iteration-loop solve time (seconds)
- `iter4::Int`: Iterations to 1e-4 tolerance
- `iter6::Int`: Iterations to 1e-6 tolerance
- `iter8::Int`: Iterations to 1e-8 tolerance
- `time4::Float64`: Time to 1e-4 tolerance
- `time6::Float64`: Time to 1e-6 tolerance
- `time8::Float64`: Time to 1e-8 tolerance

# Methods
- `is_optimal(result)`: Check if solution is optimal
"""
struct Timing
    total_time::Float64
    presolve_time::Float64
    setup_time::Float64
    scaling_time::Float64
    analyze_time::Float64
    power_iteration_time::Float64
    solve_time::Float64
end

struct Results
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
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
    timing::Timing
    interior_percentage::Float64
    reduced_activation_checks::Int
    reduced_active_iterations::Int
    reduced_first_iteration::Int
    reduced_rebuilds::Int
    reduced_build_time::Float64
    reduced_last_trigger_iteration::Int
    reduced_last_free_ratio::Float64
    reduced_last_trigger_residual::Float64
    reduced_last_trigger_sigma::Float64
    reduced_last_free_columns::Int
end

"""
Convert C results to Julia Results struct
"""
function from_c_struct(c_results::C_HPRLP_results, n::Int, m::Int)
    # Copy solution vectors
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, m)
    z = Vector{Float64}(undef, n)
    
    if c_results.x != C_NULL
        unsafe_copyto!(pointer(x), c_results.x, n)
    end
    
    if c_results.y != C_NULL
        unsafe_copyto!(pointer(y), c_results.y, m)
    end

    if c_results.z != C_NULL
        unsafe_copyto!(pointer(z), c_results.z, n)
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
        x, y, z, status,
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
        c_results.time8,
        Timing(c_results.timing.total_time,
               c_results.timing.presolve_time,
               c_results.timing.setup_time,
               c_results.timing.scaling_time,
               c_results.timing.analyze_time,
               c_results.timing.power_iteration_time,
               c_results.timing.solve_time),
        c_results.interior_percentage,
        Int(c_results.reduced_activation_checks),
        Int(c_results.reduced_active_iterations),
        Int(c_results.reduced_first_iteration),
        Int(c_results.reduced_rebuilds),
        c_results.reduced_build_time,
        Int(c_results.reduced_last_trigger_iteration),
        c_results.reduced_last_free_ratio,
        c_results.reduced_last_trigger_residual,
        c_results.reduced_last_trigger_sigma,
        Int(c_results.reduced_last_free_columns)
    )
    
    # Free C memory
    c_free_results(c_results.x, c_results.y, c_results.z)
    
    return result
end

struct BatchedResults
    x::Matrix{Float64}
    y::Matrix{Float64}
    z::Matrix{Float64}
    status::Vector{String}
    primal_obj::Vector{Float64}
    gap::Vector{Float64}
    residuals::Vector{Float64}
    iter::Vector{Int}
    time::Float64
    setup_time::Float64
    solve_time::Float64
    power_time::Float64
end

function _string_from_status_ptr(ptr::Ptr{UInt8}, k::Int)
    bytes = unsafe_wrap(Vector{UInt8}, ptr + 64 * (k - 1), 64; own=false)
    idx = findfirst(==(0x00), bytes)
    stop = idx === nothing ? 64 : idx - 1
    return String(bytes[1:stop])
end

function from_c_struct(c_results::C_HPRLP_batched_results)
    m = Int(c_results.m)
    n = Int(c_results.n)
    B = Int(c_results.batch_size)
    x = Matrix{Float64}(undef, n, B)
    y = Matrix{Float64}(undef, m, B)
    z = Matrix{Float64}(undef, n, B)
    primal_obj = Vector{Float64}(undef, B)
    gap = Vector{Float64}(undef, B)
    residuals = Vector{Float64}(undef, B)
    iter32 = Vector{Int32}(undef, B)
    if c_results.x != C_NULL; unsafe_copyto!(pointer(x), c_results.x, n * B); end
    if c_results.y != C_NULL; unsafe_copyto!(pointer(y), c_results.y, m * B); end
    if c_results.z != C_NULL; unsafe_copyto!(pointer(z), c_results.z, n * B); end
    if c_results.primal_obj != C_NULL; unsafe_copyto!(pointer(primal_obj), c_results.primal_obj, B); end
    if c_results.gap != C_NULL; unsafe_copyto!(pointer(gap), c_results.gap, B); end
    if c_results.residuals != C_NULL; unsafe_copyto!(pointer(residuals), c_results.residuals, B); end
    if c_results.iter != C_NULL; unsafe_copyto!(pointer(iter32), c_results.iter, B); end
    status = c_results.status == C_NULL ? fill("ERROR", B) : [_string_from_status_ptr(c_results.status, k) for k in 1:B]
    ref = Ref(c_results)
    c_free_batched_results(ref)
    return BatchedResults(x, y, z, status, primal_obj, gap, residuals, Int.(iter32),
                          c_results.time, c_results.setup_time, c_results.solve_time, c_results.power_time)
end

function solve_batched(model::Model,
                       C::AbstractMatrix{Float64},
                       AL::AbstractMatrix{Float64},
                       AU::AbstractMatrix{Float64},
                       l::AbstractMatrix{Float64},
                       u::AbstractMatrix{Float64},
                       params = nothing;
                       obj_constants::Union{AbstractVector{Float64}, Nothing}=nothing)
    model.ptr == C_NULL && error("Cannot solve freed model")
    B = size(C, 2)
    size(C) == (model.n, B) || error("C must have size n x B")
    size(l) == (model.n, B) || error("l must have size n x B")
    size(u) == (model.n, B) || error("u must have size n x B")
    size(AL) == (model.m, B) || error("AL must have size m x B")
    size(AU) == (model.m, B) || error("AU must have size m x B")
    obj = obj_constants === nothing ? nothing : Vector{Float64}(obj_constants)
    obj !== nothing && length(obj) != B && error("obj_constants must have length B")
    c_params = params === nothing ? nothing : to_c_struct(params)
    c_results = c_solve_batched(model.ptr, Matrix{Float64}(C), Matrix{Float64}(AL), Matrix{Float64}(AU),
                                Matrix{Float64}(l), Matrix{Float64}(u), obj, c_params)
    return from_c_struct(c_results)
end

function solve_batched(A::AbstractMatrix{Float64},
                       C::AbstractMatrix{Float64},
                       AL::AbstractMatrix{Float64},
                       AU::AbstractMatrix{Float64},
                       l::AbstractMatrix{Float64},
                       u::AbstractMatrix{Float64},
                       params = nothing;
                       obj_constants::Union{AbstractVector{Float64}, Nothing}=nothing)
    m, n = size(A)
    model = Model(A, zeros(m), zeros(m), zeros(n), zeros(n), zeros(n))
    try
        return solve_batched(model, C, AL, AU, l, u, params; obj_constants=obj_constants)
    finally
        free(model)
    end
end

is_optimal(result::BatchedResults) = all(==("OPTIMAL"), result.status)
