"""
Low-level C wrapper for HPRLP library.

This module provides direct access to the C functions and structures.
Most users should use the high-level interface in interface.jl instead.

The wrapper uses Julia's `ccall` interface which provides:
- Zero-overhead C function calls
- Automatic type conversion
- Direct memory access
- Full compatibility with C libraries

This is the standard approach for Julia-C interfacing and is used by
packages like LinearAlgebra, SparseArrays, etc.
"""

import Libdl

# Find the library path
function find_hprlp_lib()
    # Try different possible locations relative to this file
    # We're in bindings/julia/package/src/, library is in project_root/lib/
    lib_dir = abspath(joinpath(@__DIR__, "..", "..", "..", "..", "lib"))
    
    # Platform-specific library names
    if Sys.islinux()
        lib_names = ["libhprlp.so", "libhprlp.so.0"]
    elseif Sys.isapple()
        lib_names = ["libhprlp.dylib"]
    elseif Sys.iswindows()
        lib_names = ["libhprlp.dll", "hprlp.dll"]
    else
        lib_names = ["libhprlp"]
    end
    
    # Try library directory first
    for name in lib_names
        path = joinpath(lib_dir, name)
        if isfile(path)
            @info "Found HPRLP library at: $path"
            return path
        end
    end
    
    # Try system library path
    for name in lib_names
        try
            # Test if library can be loaded
            lib = Libdl.dlopen(name)
            Libdl.dlclose(lib)
            @info "Found HPRLP library in system path: $name"
            return name
        catch
            continue
        end
    end
    
    error("""
    HPRLP library not found!
    
    Please build the library first:
        cd $(abspath(joinpath(@__DIR__, "..", "..")))
        make
    
    The library should be in: $lib_dir
    """)
end

# Load library (with better error handling)
const libhprlp = find_hprlp_lib()

# Verify library was loaded successfully
function __init__()
    try
        # Test library by checking if a function exists
        lib_handle = Libdl.dlopen(libhprlp)
        Libdl.dlsym(lib_handle, :create_model_from_arrays)
        Libdl.dlsym(lib_handle, :solve)
        @info "HPRLP library loaded successfully"
    catch e
        error("Failed to load HPRLP library: $e")
    end
end

"""
    C_HPRLP_parameters

C-compatible struct for solver parameters.
Maps directly to the C struct HPRLP_parameters.
"""
mutable struct C_HPRLP_parameters
    max_iter::Int32
    stop_tol::Float64
    time_limit::Float64
    device_number::Int32
    check_iter::Int32
    use_Ruiz_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    use_bc_scaling::Bool
    
    function C_HPRLP_parameters()
        new(
            typemax(Int32),  # max_iter
            1e-4,            # stop_tol
            3600.0,          # time_limit
            0,               # device_number
            150,             # check_iter
            true,            # use_Ruiz_scaling
            true,            # use_Pock_Chambolle_scaling
            true             # use_bc_scaling
        )
    end
end

"""
    C_HPRLP_results

C-compatible struct for solver results.
Maps directly to the C struct HPRLP_results.
"""
mutable struct C_HPRLP_results
    residuals::Float64
    primal_obj::Float64
    gap::Float64
    time4::Float64
    time6::Float64
    time8::Float64
    time::Float64
    iter4::Int32
    iter6::Int32
    iter8::Int32
    iter::Int32
    status::NTuple{64, UInt8}  # C char array[64]
    x::Ptr{Float64}            # Primal solution array
    y::Ptr{Float64}            # Dual solution array
end

"""
Call C function: create_model_from_arrays

Create an LP model from raw arrays.
Returns a pointer to LP_info_cpu structure.
"""
function c_create_model_from_arrays(m::Int, n::Int, nnz::Int,
                                     rowPtr::Vector{Int32}, colIndex::Vector{Int32}, values::Vector{Float64},
                                     AL::Vector{Float64}, AU::Vector{Float64},
                                     l::Vector{Float64}, u::Vector{Float64},
                                     c::Vector{Float64},
                                     is_csc::Bool)
    
    model_ptr = ccall((:create_model_from_arrays, libhprlp), Ptr{Cvoid},
                      (Int32, Int32, Int32,
                       Ptr{Int32}, Ptr{Int32}, Ptr{Float64},
                       Ptr{Float64}, Ptr{Float64},
                       Ptr{Float64}, Ptr{Float64},
                       Ptr{Float64},
                       Bool),
                      m, n, nnz,
                      rowPtr, colIndex, values,
                      AL, AU,
                      l, u,
                      c,
                      is_csc)
    
    return model_ptr
end

"""
Call C function: create_model_from_mps

Create an LP model from MPS file.
Returns a pointer to LP_info_cpu structure.
"""
function c_create_model_from_mps(filename::String)
    model_ptr = ccall((:create_model_from_mps, libhprlp), Ptr{Cvoid},
                      (Cstring,),
                      filename)
    return model_ptr
end

"""
Call C function: solve

Solve an LP model.
"""
function c_solve_model(model_ptr::Ptr{Cvoid}, param::Union{C_HPRLP_parameters, Nothing})
    if param === nothing
        # Pass NULL for default parameters
        result = ccall((:solve, libhprlp), C_HPRLP_results,
                       (Ptr{Cvoid}, Ptr{Cvoid}),
                       model_ptr, C_NULL)
    else
        result = ccall((:solve, libhprlp), C_HPRLP_results,
                       (Ptr{Cvoid}, Ref{C_HPRLP_parameters}),
                       model_ptr, Ref(param))
    end
    return result
end

"""
Call C function: free_model

Free an LP model and release memory.
"""
function c_free_model(model_ptr::Ptr{Cvoid})
    if model_ptr != C_NULL
        ccall((:free_model, libhprlp), Cvoid,
              (Ptr{Cvoid},),
              model_ptr)
    end
end

"""
Free memory allocated by C library for solution vectors.

Note: This uses the system free() function which is compatible
with the malloc() used in the C library.
"""
function c_free_results(x_ptr::Ptr{Float64}, y_ptr::Ptr{Float64})
    if x_ptr != C_NULL
        # Use system free() which matches malloc() in C library
        ccall(:free, Cvoid, (Ptr{Float64},), x_ptr)
    end
    if y_ptr != C_NULL
        ccall(:free, Cvoid, (Ptr{Float64},), y_ptr)
    end
end
