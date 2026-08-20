#!/usr/bin/env julia

"""
Example: Solve LP from MPS File with HPRLP.jl

This example demonstrates solving an LP problem from an MPS format file
using the Model-based API. Shows model reuse with different parameters.
"""

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using HPRLP

println("=" ^ 70)
println("HPRLP.jl Example: Solve from MPS File")
println("=" ^ 70)
println()

# Find MPS file (try a few common locations)
global mps_file = nothing
possible_paths = [
    joinpath(@__DIR__, "..", "..", "..", "data", "model.mps"),
]

for path in possible_paths
    if isfile(path)
        global mps_file = path
        break
    end
end

if mps_file === nothing
    println("ERROR: Could not find model.mps file")
    println("Tried locations:")
    for path in possible_paths
        println("  - ", abspath(path))
    end
    println()
    println("Please provide path to an MPS file as command line argument:")
    println("  julia example_mps_file.jl <path_to_mps_file>")
    exit(1)
end

# Check for command line argument
if length(ARGS) > 0
    global mps_file = ARGS[1]
    if !isfile(mps_file)
        println("ERROR: File not found: ", mps_file)
        exit(1)
    end
end

println("MPS file: ", mps_file)
println()

# Create model from MPS file
println("Creating model from MPS file...")
model = Model(mps_file)
println("Model created: $(model.m) constraints, $(model.n) variables")
println()

# Solve with custom parameters
params1 = Parameters(
    stop_tol = 1e-9,
    device_number = 0
)

println("Solve with custom parameters:")
println()

result1 = solve(model, params1)

# Display results
println("=" ^ 70)
println("Solution Results")
println("=" ^ 70)
println()

println("Status:         ", result1.status)
println("Objective:      ", round(result1.primal_obj, digits=6))
println("Iterations:     ", result1.iter)
println("Time:           ", round(result1.time, digits=4), " seconds")
println("Duality gap:    ", result1.gap)
println("Residuals:      ", result1.residuals)
println()
