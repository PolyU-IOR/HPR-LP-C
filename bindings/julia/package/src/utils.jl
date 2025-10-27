"""
Utility functions for HPRLP.jl
"""

"""
    is_optimal(result::Results)

Check if the solver found an optimal solution.

# Arguments
- `result`: Results object from solve() or solve_mps()

# Returns
- `true` if status is "OPTIMAL", `false` otherwise

# Example
```julia
result = solve(A, AL, AU, l, u, c)
if is_optimal(result)
    println("Found optimal solution!")
end
```
"""
function is_optimal(result::Results)
    return result.status == "OPTIMAL"
end

"""
    get_mps_dimensions(filename::String)

Parse MPS file to extract problem dimensions (m, n).
This is a simple parser that reads the MPS file format.

# Arguments
- `filename`: Path to MPS file

# Returns
- `(m, n)`: Number of constraints and variables
"""
function get_mps_dimensions(filename::String)
    m = 0  # constraints
    n = 0  # variables
    
    in_rows = false
    in_columns = false
    
    open(filename, "r") do file
        for line in eachline(file)
            line = strip(line)
            
            # Skip comments and empty lines
            if isempty(line) || startswith(line, "*")
                continue
            end
            
            # Check sections
            if startswith(line, "ROWS")
                in_rows = true
                in_columns = false
                continue
            elseif startswith(line, "COLUMNS")
                in_rows = false
                in_columns = true
                continue
            elseif startswith(line, "RHS") || startswith(line, "BOUNDS") || 
                   startswith(line, "RANGES") || startswith(line, "ENDATA")
                in_rows = false
                in_columns = false
                continue
            end
            
            # Count rows (skip objective row which starts with N)
            if in_rows
                parts = split(line)
                if length(parts) >= 2 && parts[1] != "N"
                    m += 1
                end
            end
            
            # Count columns (unique variable names)
            if in_columns
                parts = split(line)
                if length(parts) >= 3
                    # First column is variable name
                    # We'll count unique variables by tracking them
                    # For simplicity, we'll estimate from the file
                end
            end
        end
    end
    
    # For variables, we need a more sophisticated approach
    # Let's count unique variable names
    variables = Set{String}()
    in_columns = false
    
    open(filename, "r") do file
        for line in eachline(file)
            line = strip(line)
            
            if startswith(line, "COLUMNS")
                in_columns = true
                continue
            elseif startswith(line, "RHS") || startswith(line, "BOUNDS")
                break
            end
            
            if in_columns && !isempty(line) && !startswith(line, "*")
                parts = split(line)
                if length(parts) >= 3
                    push!(variables, parts[1])
                end
            end
        end
    end
    
    n = length(variables)
    
    return (m, n)
end

"""
    Base.show(io::IO, result::Results)

Pretty-print Results object.
"""
function Base.show(io::IO, result::Results)
    println(io, "HPRLP Results")
    println(io, "═" ^ 50)
    println(io, "Status:        ", result.status)
    println(io, "Primal obj:    ", result.primal_obj)
    println(io, "Gap:           ", result.gap)
    println(io, "Residuals:     ", result.residuals)
    println(io, "Iterations:    ", result.iter)
    println(io, "Time (s):      ", result.time)
    println(io)
    println(io, "Convergence milestones:")
    println(io, "  1e-4:  iter=", result.iter4, ", time=", result.time4, "s")
    println(io, "  1e-6:  iter=", result.iter6, ", time=", result.time6, "s")
    println(io, "  1e-8:  iter=", result.iter8, ", time=", result.time8, "s")
    println(io)
    println(io, "Solution vectors:")
    println(io, "  x (primal): ", length(result.x), " elements")
    println(io, "  y (dual):   ", length(result.y), " elements")
end

"""
    Base.show(io::IO, params::Parameters)

Pretty-print Parameters object.
"""
function Base.show(io::IO, params::Parameters)
    println(io, "HPRLP Parameters")
    println(io, "═" ^ 50)
    println(io, "max_iter:                  ", params.max_iter)
    println(io, "stop_tol:                  ", params.stop_tol)
    println(io, "time_limit:                ", params.time_limit, " s")
    println(io, "device_number:             ", params.device_number)
    println(io, "check_iter:                ", params.check_iter)
    println(io, "use_Ruiz_scaling:          ", params.use_Ruiz_scaling)
    println(io, "use_Pock_Chambolle_scaling:", params.use_Pock_Chambolle_scaling)
    println(io, "use_bc_scaling:            ", params.use_bc_scaling)
end

# Define the product of sets at module level (required for struct definition)
MOI.Utilities.@product_of_sets(
    LPSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
)

# Define the cache type with MatrixOfConstraints (same approach as Clp.jl)
const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Float64},
        LPSets{Float64},
    },
}

# Helper function to extract LP data from JuMP model
# This uses MOI's MatrixOfConstraints for efficient matrix extraction
function extract_lp_data(model::JuMP.Model)
    # Get the backend MOI model
    moi_backend = backend(model)
    
    # Extract objective function from JuMP model
    obj = MOI.get(moi_backend, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    sense = MOI.get(moi_backend, MOI.ObjectiveSense())
    
    # Create cache and copy model for constraint matrix
    cache = OptimizerCache()
    MOI.copy_to(cache, moi_backend)
    
    # Extract sparse matrix and bounds from cache
    A = cache.constraints.coefficients
    row_bounds = cache.constraints.constants
    
    # Convert bounds from sets to vectors
    AL = row_bounds.lower
    AU = row_bounds.upper
    
    # Extract variable bounds
    l = cache.variables.lower
    u = cache.variables.upper
    
    # Convert objective to coefficient vector using sparse vector
    indices = [term.variable.value for term in obj.terms]
    values = [term.coefficient for term in obj.terms]
    c = Vector(sparsevec(indices, values, A.n)) 
    obj_constant = obj.constant
    
    # Handle maximization
    if sense == MOI.MAX_SENSE
        c = -c
        obj_constant = -obj_constant
    end
    
    # Convert to standard Julia SparseMatrixCSC
    A_sparse = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval)
    
    return A_sparse, AL, AU, l, u, c, obj_constant
end

"""
    solve(jump_model::JuMP.Model, params::Parameters = Parameters())

Solve a JuMP model using HPRLP.

Extracts the LP data from a JuMP model and solves it using HPRLP's solver.

# Arguments
- `jump_model`: JuMP Model object containing the LP problem
- `params`: HPRLP Parameters object (optional, uses defaults if not provided)

# Returns
- `Results` object containing solution and statistics

# Example
```julia
using JuMP
using HPRLP

# Create JuMP model
jump_model = JuMP.Model()
@variable(jump_model, x >= 0)
@variable(jump_model, y >= 0)
@objective(jump_model, Min, -3x - 5y)
@constraint(jump_model, x + 2y <= 10)
@constraint(jump_model, 3x + y <= 12)

# Solve with HPRLP
params = Parameters(stop_tol = 1e-9)
result = solve(jump_model, params)
println("Optimal value: ", result.primal_obj)
```
"""
function solve(jump_model::JuMP.Model, params::Parameters = Parameters())
    # Extract LP data from JuMP model
    A, AL, AU, l, u, c, obj_constant = extract_lp_data(jump_model)
    
    # Create HPRLP model from the extracted data
    hprlp_model = Model(A, AL, AU, l, u, c; obj_constant = obj_constant)
    
    # Solve using HPRLP
    result = solve(hprlp_model, params)
    
    return result
end