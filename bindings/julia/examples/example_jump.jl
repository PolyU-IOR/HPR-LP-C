#!/usr/bin/env julia

"""
Example: Solve JuMP Model with HPRLP.jl

This example demonstrates solving an LP problem created with JuMP
and solved using HPRLP's high-performance solver.

Problem:
    minimize    -3x₁ - 5x₂
    subject to  x₁ + 2x₂ <= 10
                3x₁ + x₂ <= 12
                x₁, x₂ >= 0

Expected solution: x₁ ≈ 2.8, x₂ ≈ 3.6, obj ≈ -26.4
"""

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using HPRLP
using JuMP

println("=" ^ 70)
println("HPRLP.jl Example: Solve JuMP Model")
println("=" ^ 70)
println()

# Create JuMP model
println("Creating JuMP model...")
jump_model = JuMP.Model()

function simple_example(model)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    @objective(model, Min, -3x1 - 5x2)

    @constraint(model, 1x1 + 2x2 <= 10)
    @constraint(model, 3x1 + 1x2 <= 12)
end

# For more examples, please refer to the JuMP documentation: https://jump.dev/JuMP.jl/stable/tutorials/linear/introduction/
simple_example(jump_model)

println("Problem:")
println("  minimize    -3x₁ - 5x₂")
println("  subject to  x₁ + 2x₂ <= 10")
println("              3x₁ + x₂ <= 12")
println("              x₁, x₂ >= 0")
println()

# Configure solver parameters
println("Configuring HPRLP parameters...")
params = Parameters(
    stop_tol = 1e-9,
    device_number = 0
)

# Solve with HPRLP
println("Solving with HPRLP...")
println()

result = solve(jump_model, params)

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
    
    # Check if solution is feasible and optimal
    tol = 1e-4
    if x[1] + 2*x[2] <= 10 + tol && 
       3*x[1] + x[2] <= 12 + tol &&
       x[1] >= -tol && x[2] >= -tol
        println("✓ Solution is feasible!")
    else
        println("✗ Solution may not be feasible")
    end
else
    println("⚠ Did not reach optimal solution")
    println("  Try increasing max_iter or time_limit")
end
