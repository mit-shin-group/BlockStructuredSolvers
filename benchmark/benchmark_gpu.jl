using LinearAlgebra, SparseArrays, BlockArrays
using CUDA
using AMDGPU
using Random
using CUDSS
using BlockStructuredSolvers

import Pkg
include("utils.jl")

using Printf, ProgressBars, Statistics, Dates

if CUDA.functional()
    device!(0)
end

######

seed = 42 # random seed for reproducibility
T = Float64

# Determine GPU backend
if AMDGPU.functional()
    M = ROCArray
else
    M = CuArray
end

# Helper functions for cleaner code
function gpu_elapsed(f, backend_type)
    if backend_type == CuArray
        return CUDA.@elapsed f()
    else
        return AMDGPU.@elapsed f()
    end
end

function gpu_sync(f, backend_type)
    if backend_type == CuArray
        return CUDA.@sync f()
    else
        return AMDGPU.@sync f()
    end
end

function gpu_allowscalar(f, backend_type)
    if backend_type == CuArray
        @CUDA.allowscalar f()
    else
        @AMDGPU.allowscalar f()
    end
end

function gpu_reclaim(backend_type)
    GC.gc()
    if backend_type == CuArray
        CUDA.reclaim()
    else
        AMDGPU.GC.gc()
    end
end

function fill_gpu_tensors!(A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, A_list_gpu, B_list_gpu, d_list_gpu, N, backend_type)
    gpu_allowscalar(backend_type) do
        for i in 1:N
            A_tensor_gpu[:, :, i] = A_list_gpu[i]
            d_tensor_gpu[:, :, i] = d_list_gpu[i]
        end
        for i in 1:N-1
            B_tensor_gpu[:, :, i] = B_list_gpu[i]
        end
    end
end

function benchmark_cudss(BigMatrix, d, N, n, backend_type)
    # Only run CUDSS if CUDA is available
    if !(CUDA.functional() && backend_type == CuArray)
        return nothing
    end
    
    # Initialization time
    init_time = gpu_elapsed(backend_type) do
        BigMatrix_cudss = CUSPARSE.CuSparseMatrixCSR(BigMatrix)
        BigMatrix_cudss = CUDSS.CudssMatrix(BigMatrix_cudss, "SPD", 'U')
        config = CUDSS.CudssConfig()
        data = CUDSS.CudssData()
        solver = CUDSS.CudssSolver(BigMatrix_cudss, config, data)
        T = eltype(BigMatrix)
        d_cudss = CuVector{T}(vec(d))
        x_cudss = CUDA.zeros(T, N*n)
        b_cudss = CUDA.zeros(T, N*n)
        CUDSS.cudss("analysis", solver, x_cudss, b_cudss)
    end
    
    # Factorization and solve
    factor_time = gpu_elapsed(backend_type) do
        CUDSS.cudss("factorization", solver, x_cudss, b_cudss)
    end
    
    solve_time = gpu_elapsed(backend_type) do
        CUDSS.cudss("solve", solver, x_cudss, d_cudss)
    end
    
    solution = Array(x_cudss[:, 1])
    
    # Cleanup
    x_cudss = nothing
    d_cudss = nothing
    b_cudss = nothing
    BigMatrix_cudss = nothing
    solver = nothing
    config = nothing
    data = nothing
    gpu_reclaim(backend_type)
    
    return (init_time, factor_time, solve_time), solution
end

function benchmark_gpu_solver(solver_type, A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, N, n, backend_type)
    # Measure initialization time properly by timing everything
    start_time = time()
    
    if solver_type == :batched
        data = initialize_batched(N, n, T, backend_type)
    else  # sequential
        data = initialize_seq(N, n, T, backend_type)
    end
    
    copyto!(data.A_tensor, A_tensor_gpu)
    copyto!(data.B_tensor, B_tensor_gpu)
    copyto!(data.d_tensor, d_tensor_gpu)
    
    # Synchronize to ensure all operations are complete
    if backend_type == CuArray
        CUDA.synchronize()
    else
        AMDGPU.synchronize()
    end
    
    init_time = time() - start_time
    
    # Factorization
    factor_time = gpu_elapsed(backend_type) do
        gpu_sync(() -> factorize!(data), backend_type)
    end
    
    # Solve
    solve_time = gpu_elapsed(backend_type) do
        gpu_sync(() -> solve!(data), backend_type)
    end
    
    # Extract solution
    solution = vcat([vec(Array(data.d_list[i])) for i in 1:N]...)
    
    return (init_time, factor_time, solve_time), solution
end

function compute_residual(solution, A_list, B_list, d_list, N)
    # Convert solution back to block structure
    n = length(d_list[1])
    x_blocks = [solution[(i-1)*n+1:i*n] for i in 1:N]
    
    # Compute Ax using block structure
    Ax_blocks = Vector{Vector{T}}(undef, N)
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

function run_gpu(N, n)
    # Set random seed for reproducibility
    Random.seed!(seed)

    # Generate data
    A_list, B_list, x_list, x, d_list = generate_data(N, n)
    if CUDA.functional()
        A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu = to_nvidia_gpu(A_list, B_list, x_list, x, d_list)
    else
        A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu = to_amd_gpu(A_list, B_list, x_list, x, d_list)
    end
    BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)

    # Storage for results
    solutions = Dict{String, Vector{T}}()
    residuals = Dict{String, T}()
    timing_results = Dict{String, Tuple{T, T, T}}()

    # Create GPU tensors
    A_tensor_gpu = M{T}(undef, n, n, N)
    B_tensor_gpu = M{T}(undef, n, n, N-1)
    d_tensor_gpu = M{T}(undef, n, 1, N)
    
    # Fill GPU tensors
    fill_gpu_tensors!(A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, A_list_gpu, B_list_gpu, d_list_gpu, N, M)

    # Benchmark CUDSS
    cudss_result = benchmark_cudss(BigMatrix, d, N, n, M)
    if cudss_result !== nothing
        timing_results["CUDSS"], solutions["CUDSS"] = cudss_result
    end

    # Benchmark GPU solvers
    timing_results["GPU_Batched"], solutions["GPU_Batched"] = 
        benchmark_gpu_solver(:batched, A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, N, n, M)
    
    timing_results["GPU_Sequential"], solutions["GPU_Sequential"] = 
        benchmark_gpu_solver(:sequential, A_tensor_gpu, B_tensor_gpu, d_tensor_gpu, N, n, M)

    # Compute residuals
    for (solver_name, solution) in solutions
        residuals[solver_name] = compute_residual(solution, A_list, B_list, d_list, N)
    end

    # Cleanup
    gpu_reclaim(M)

    return timing_results, residuals
end

function write_results_to_file(io, solver_name, timing_data, residual_data)
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

function run_benchmark_suite(problem_sizes, iterations=10, output_file="gpu_benchmark_results.txt")
    """
    Run benchmarks for multiple problem sizes and save results to file.
    """
    
    all_results = []
    
    # Single warmup at the beginning
    println("Running warmup...")
    _, _ = run_gpu(20, 32)
    println("Warmup completed.\n")
    
    # Open output file in append mode
    open(output_file, "a") do io
        # Write header
        println(io, "\n" * "="^80)
        println(io, "NEW BENCHMARK RUN")
        println(io, "GPU Block Structured Solver Benchmark Results")
        println(io, "Generated: $(now())")
        println(io, "GPU Backend: $(M == CuArray ? "CUDA" : "ROCm")")
        println(io, "Iterations per problem size: $iterations")
        println(io, "="^80)
        println(io)
        
        for (N, n) in problem_sizes
            println("Running benchmark for N=$N, n=$n...")
            println(io, "Problem Size: N=$N, n=$n (Total size: $(N*n))")
            println(io, "-"^50)
            
            # Storage for this problem size
            all_timing_results = Dict{String, Vector{Tuple{T, T, T}}}()
            all_residuals = Dict{String, Vector{T}}()
            
            println("  Running $iterations iterations...")
            for i = 1:iterations
                timing_results, residuals = run_gpu(N, n)
                
                # Collect results
                for (solver_name, timing) in timing_results
                    if !haskey(all_timing_results, solver_name)
                        all_timing_results[solver_name] = Tuple{T, T, T}[]
                    end
                    push!(all_timing_results[solver_name], timing)
                end
                
                for (solver_name, residual) in residuals
                    if !haskey(all_residuals, solver_name)
                        all_residuals[solver_name] = T[]
                    end
                    push!(all_residuals[solver_name], residual)
                end
            end
            
            # Write results for this problem size
            println(io, "Results:")
            for solver_name in sort(collect(keys(all_timing_results)))
                write_results_to_file(io, solver_name, all_timing_results[solver_name], all_residuals[solver_name])
            end
            
            # Store for summary
            push!(all_results, (N, n, all_timing_results, all_residuals))
            println(io, "="^50 * "\n")
        end
        
        # Write summary table
        println(io, "SUMMARY TABLE")
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
    
    println("Results saved to: $output_file")
end

# Example usage - modify these problem sizes as needed
problem_sizes = [
    (256, 1024),
    (512, 512),
    (1024, 256),
    (2048, 128),
    (4096, 64),
    (8192, 32),
]

# Run the benchmark suite
run_benchmark_suite(problem_sizes, 10, "gpu_benchmark_results.txt")
