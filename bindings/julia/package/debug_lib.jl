#!/usr/bin/env julia
# Quick test to debug the ccall issue

println("Testing HPRLP library loading...")

# Test 1: Can we find and load the library?
import Libdl

lib_path = "/home/chenkaihuang/workspace/programming/HPR-LP-C/lib/libhprlp.so"

if !isfile(lib_path)
    println("ERROR: Library not found at $lib_path")
    exit(1)
end

println("✓ Library file exists")

# Test 2: Can we load it?
try
    lib = Libdl.dlopen(lib_path)
    println("✓ Library loaded successfully")
    
    # Test 3: Can we find the solve_lp symbol?
    try
        sym = Libdl.dlsym(lib, :solve_lp)
        println("✓ Found solve_lp symbol at: $sym")
    catch e
        println("✗ Could not find solve_lp symbol: $e")
    end
    
    Libdl.dlclose(lib)
catch e
    println("✗ Failed to load library: $e")
    exit(1)
end

println("")
println("All basic tests passed!")
println("The issue is likely with the C++ std::string in the result struct.")
println("We need to handle this differently in Julia.")
