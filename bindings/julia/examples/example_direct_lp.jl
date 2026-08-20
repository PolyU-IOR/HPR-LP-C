#!/usr/bin/env julia

"""
Example: Direct LP Data Entry with HPRLP.jl

This example demonstrates solving an LP problem by directly providing
constraint matrices and vectors using the Model-based API.

Problem from model.mps:
    minimize    -3x₁ - 5x₂
    subject to  x₁ + 2x₂ <= 10
                3x₁ + x₂ <= 12
                x₁, x₂ >= 0

Expected solution: x₁ ≈ 2.8, x₂ ≈ 3.6, obj ≈ -26.4
"""

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using HPRLP
using SparseArrays
using LinearAlgebra

println("=" ^ 70)
println("HPRLP.jl Example: Direct LP Input")
println("=" ^ 70)
println()

println("Problem:")
println("  minimize    -3x₁ - 5x₂")
println("  subject to  x₁ + 2x₂ <= 10")
println("              3x₁ + x₂ <= 12")
println("              x₁, x₂ >= 0")
println()

# Constraint matrix A (sparse format)
# Row 0: x₁ + 2x₂ <= 10
# Row 1: 3x₁ + x₂ <= 12
A = sparse([1.0 2.0;
            3.0 1.0])

println("Constraint matrix A:")
println(Matrix(A))  # Display as dense for readability
println()

# Constraint bounds: AL <= Ax <= AU
AL = [-Inf, -Inf]  # No lower bounds
AU = [10.0, 12.0]   # Upper bounds (RHS values)

# Variable bounds: l <= x <= u
l = [0.0, 0.0]      # Lower bounds (x >= 0)
u = [Inf, Inf]      # No upper bounds

# Objective: minimize c'x
c = [-3.0, -5.0]

# Create model
println("Creating model...")
model = Model(A, AL, AU, l, u, c)
println("Model created: $(model.m) constraints, $(model.n) variables")
println()

# Configure solver parameters
println("Configuring solver parameters...")
params = Parameters(
    stop_tol = 1e-9,
    device_number = 0
)
println()

# Solve with parameters
println("Solving with parameters (stop_tol=1e-9)...")
println()

result = solve(model, params)

# Display results
println("=" ^ 70)
println("Solution Results")
println("=" ^ 70)
println()

println("Status:         ", result.status)
println("Objective:      ", round(result.primal_obj, digits=6))
println("Solution:       x = [", join(round.(result.x, digits=6), ", "), "]")
println("Iterations:     ", result.iter)
println("Time:           ", round(result.time, digits=4), " seconds")
println("Duality gap:    ", result.gap)
println("Residuals:      ", result.residuals)
println()

# Verify solution
if result.status == "OPTIMAL"
    x = result.x
    println("Verification:")
    println("  Constraint 1:  x₁ + 2x₂ = ", round(x[1] + 2*x[2], digits=6), " <= 10")
    println("  Constraint 2:  3x₁ + x₂ = ", round(3*x[1] + x[2], digits=6), " <= 12")
    println("  Objective:     -3x₁ - 5x₂ = ", round(-3*x[1] - 5*x[2], digits=6))
    println()
    println("✓ Solution is optimal!")
else
    println("⚠ Solution is not optimal!")
end

# Note: Model is automatically freed when garbage collected
# Or explicitly: free(model)

println()
println("=" ^ 70)
