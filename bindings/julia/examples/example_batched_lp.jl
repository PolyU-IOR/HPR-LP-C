#!/usr/bin/env julia

# Example: batched LP solves sharing one sparse matrix A.
# Each column of C, AL, AU, l, and u defines one LP instance.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "package"))

using HPRLP
using SparseArrays

println("=" ^ 70)
println("HPRLP.jl Example: Batched Shared-A LP")
println("=" ^ 70)

A = sparse([1.0 2.0;
            3.0 1.0])
B = 3

C = [-3.0 -2.0 -4.0;
     -5.0 -6.0 -4.0]
AL = fill(-Inf, 2, B)
AU = [10.0 9.0 11.0;
      12.0 13.0 11.0]
l = zeros(2, B)
u = fill(Inf, 2, B)
u[1, 3] = 4.0

params = Parameters(stop_tol = 1e-8, max_iter = 200000, use_presolve = false)
result = solve_batched(A, C, AL, AU, l, u, params)

println("Batch size: ", length(result.status))
println("Total time: ", round(result.time, digits=4), " seconds")
for k in eachindex(result.status)
    println("[$(k)] status=$(result.status[k]) obj=$(result.primal_obj[k]) residual=$(result.residuals[k]) iter=$(result.iter[k]) x=$(result.x[:, k])")
end
