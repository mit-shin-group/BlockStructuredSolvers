using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse
using Random
using LDLFactorizations
using HSL
using MKL
using BlockStructuredSolvers

import Pkg
include("utils.jl")

using Printf, ProgressBars, Statistics, Dates

######

N = 128
n = 128 # size of each block
seed = 42 # random seed for reproducibility
T = Float64

println("BLAS Configuration:")
println(LinearAlgebra.BLAS.get_config())
println()

# Helper functions for CPU solvers
function benchmark_ma57(BigMatrix, d, N, n)
    try
        # Initialization time
        start_time = time()
        BigMatrix_57 = Ma57(BigMatrix)
        d_57 = deepcopy(d)
        init_time = time() - start_time
        
        # Factorization
        factor_time = @elapsed ma57_factorize!(BigMatrix_57)
        
        # Solve
        solve_time = @elapsed ma57_solve(BigMatrix_57, d_57)
        
        solution = d_57
        
        # Cleanup
        BigMatrix_57 = nothing
        d_57 = nothing
        GC.gc()
        
        return (init_time, factor_time, solve_time), vec(solution)
    catch e
        println("Warning: MA57 not available or failed: ", e)
        return nothing
    end
end

function benchmark_cholmod(BigMatrix, d, N, n)
    try
        # CHOLMOD has no initialization phase - direct operation on matrix
        init_time = 0.0
        
        # Prepare data copies
        BigMatrix_copy = deepcopy(BigMatrix)
        d_copy = deepcopy(d)
        
        # Factorization
        factor_time = @elapsed F = cholesky(BigMatrix_copy)
        
        # Solve
        solve_time = @elapsed solution = F \ d_copy
        
        # Cleanup
        F = nothing
        GC.gc()
        
        return (init_time, factor_time, solve_time), vec(solution)
    catch e
        println("Warning: CHOLMOD failed: ", e)
        return nothing
    end
end

function benchmark_ldl(BigMatrix, d, N, n)
    try
        # LDL has no initialization phase - direct operation on matrix
        init_time = 0.0
        
        # Prepare data copies
        BigMatrix_copy = deepcopy(BigMatrix)
        d_copy = deepcopy(d)
        P = Vector(1:N*n)  # Create permutation vector
        
        # Factorization
        factor_time = @elapsed LDLT = ldl(BigMatrix_copy; P=P)
        
        # Solve
        solve_time = @elapsed solution = LDLT \ d_copy
        
        # Cleanup
        LDLT = nothing
        GC.gc()
        
        return (init_time, factor_time, solve_time), vec(solution)
    catch e
        println("Warning: LDL failed: ", e)
        return nothing
    end
end



function compute_residual_cpu(solution, A_list, B_list, d_list, N)
    # Convert solution back to block structure
    n = length(d_list[1])
    x_blocks = [solution[(i-1)*n+1:i*n] for i in 1:N]
    
    # Compute Ax using block structure
    Ax_blocks = Vector{Vector{Float64}}(undef, N)
    Ax_blocks[1] = A_list[1] * x_blocks[1] + B_list[1] * x_blocks[2]
    for i = 2:N-1
        Ax_blocks[i] = B_list[i-1]' * x_blocks[i-1] + A_list[i] * x_blocks[i] + B_list[i] * x_blocks[i+1]
    end
    Ax_blocks[N] = B_list[N-1]' * x_blocks[N-1] + A_list[N] * x_blocks[N]
    
    # Convert to full vectors and compute residual
    Ax_full = vcat(Ax_blocks...)
    d_full = vcat([d_list[i][:] for i in 1:N]...)
    return norm(Ax_full - d_full, 2)
end

function run_cpu(N, n)
    # Set random seed for reproducibility
    Random.seed!(seed)

    # Generate data
    A_list, B_list, x_list, x, d_list = generate_data(N, n)
    BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)

    # Storage for results
    solutions = Dict{String, Vector{Float64}}()
    residuals = Dict{String, Float64}()
    timing_results = Dict{String, Tuple{Float64, Float64, Float64}}()

    # Benchmark MA57
    ma57_result = benchmark_ma57(BigMatrix, d, N, n)
    if ma57_result !== nothing
        timing_results["MA57"], solutions["MA57"] = ma57_result
    end

    # Benchmark CHOLMOD
    cholmod_result = benchmark_cholmod(BigMatrix, d, N, n)
    if cholmod_result !== nothing
        timing_results["CHOLMOD"], solutions["CHOLMOD"] = cholmod_result
    end

    # Benchmark LDL
    ldl_result = benchmark_ldl(BigMatrix, d, N, n)
    if ldl_result !== nothing
        timing_results["LDL"], solutions["LDL"] = ldl_result
    end



    # Compute residuals
    for (solver_name, solution) in solutions
        residuals[solver_name] = compute_residual_cpu(solution, A_list, B_list, d_list, N)
    end

    # Cleanup
    GC.gc()

    return timing_results, residuals
end

function write_cpu_results_to_file(io, solver_name, timing_data, residual_data)
    init_times = [t[1] for t in timing_data]
    factor_times = [t[2] for t in timing_data]
    solve_times = [t[3] for t in timing_data]
    total_times = factor_times .+ solve_times
    
    println(io, @sprintf("  %s:", solver_name))
    println(io, @sprintf("    Init: %.6f ± %.6f ms", mean(init_times)*1000, std(init_times)*1000))
    println(io, @sprintf("    Factorize: %.6f ± %.6f ms", mean(factor_times)*1000, std(factor_times)*1000))
    println(io, @sprintf("    Solve: %.6f ± %.6f ms", mean(solve_times)*1000, std(solve_times)*1000))
    println(io, @sprintf("    Total: %.6f ± %.6f ms", mean(total_times)*1000, std(total_times)*1000))
    println(io, @sprintf("    Residual: %.2e ± %.2e", mean(residual_data), std(residual_data)))
    println(io)
end

function run_cpu_benchmark_suite(problem_sizes, iterations=10, output_file="cpu_benchmark_results.txt")
    """
    Run CPU benchmarks for multiple problem sizes and save results to file.
    """
    
    all_results = []
    
    # Single warmup at the beginning
    println("Running CPU warmup...")
    _, _ = run_cpu(20, 32)
    println("CPU warmup completed.\n")
    
    # Open output file in append mode
    open(output_file, "a") do io
        # Write header
        println(io, "\n" * "="^80)
        println(io, "NEW CPU BENCHMARK RUN")
        println(io, "CPU Block Structured Solver Benchmark Results")
        println(io, "Generated: $(now())")
        println(io, "BLAS: $(LinearAlgebra.BLAS.get_config().loaded_libs[1].libname)")
        println(io, "Iterations per problem size: $iterations")
        println(io, "="^80)
        println(io)
        
        for (N, n) in problem_sizes
            println("Running CPU benchmark for N=$N, n=$n...")
            println(io, "Problem Size: N=$N, n=$n (Total size: $(N*n))")
            println(io, "-"^50)
            
            # Storage for this problem size
            all_timing_results = Dict{String, Vector{Tuple{Float64, Float64, Float64}}}()
            all_residuals = Dict{String, Vector{Float64}}()
            
            println("  Running $iterations iterations...")
            for i = 1:iterations
                timing_results, residuals = run_cpu(N, n)
                
                # Collect results
                for (solver_name, timing) in timing_results
                    if !haskey(all_timing_results, solver_name)
                        all_timing_results[solver_name] = Tuple{Float64, Float64, Float64}[]
                    end
                    push!(all_timing_results[solver_name], timing)
                end
                
                for (solver_name, residual) in residuals
                    if !haskey(all_residuals, solver_name)
                        all_residuals[solver_name] = Float64[]
                    end
                    push!(all_residuals[solver_name], residual)
                end
            end
            
            # Write results for this problem size
            println(io, "Results:")
            for solver_name in sort(collect(keys(all_timing_results)))
                write_cpu_results_to_file(io, solver_name, all_timing_results[solver_name], all_residuals[solver_name])
            end
            
            # Store for summary
            push!(all_results, (N, n, all_timing_results, all_residuals))
            println(io, "="^50 * "\n")
        end
        
        # Write summary table
        println(io, "CPU SUMMARY TABLE")
        println(io, "="^120)
        println(io, @sprintf("%-8s %-8s %-12s %-15s %-12s %-12s %-12s %-12s", 
               "N", "n", "Total Size", "Solver", "Init (ms)", "Factor (ms)", "Solve (ms)", "Total (ms)"))
        println(io, "-"^120)
        
        for (N, n, timing_results, residuals) in all_results
            total_size = N * n
            for solver_name in sort(collect(keys(timing_results)))
                timing_data = timing_results[solver_name]
                init_times = [t[1] for t in timing_data]
                factor_times = [t[2] for t in timing_data]
                solve_times = [t[3] for t in timing_data]
                total_times = factor_times .+ solve_times
                println(io, @sprintf("%-8d %-8d %-12d %-15s %-12.6f %-12.6f %-12.6f %-12.6f", 
                       N, n, total_size, solver_name, mean(init_times)*1000, 
                       mean(factor_times)*1000, mean(solve_times)*1000, mean(total_times)*1000))
            end
        end
    end
    
    println("CPU results saved to: $output_file")
end

# Example usage - modify these problem sizes as needed
cpu_problem_sizes = [
    (32, 32),
    (64, 64),
    (128, 128),
]

# Run the CPU benchmark suite
run_cpu_benchmark_suite(cpu_problem_sizes, 10, "cpu_benchmark_results.txt") 