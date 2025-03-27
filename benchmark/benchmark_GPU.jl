using BlockStructuredSolvers
using CUDA
using BenchmarkTools
using LinearAlgebra
using Statistics
using Printf
using Random
using ProgressBars

include("utils.jl")

function run(m, n, P_start, level)
    # Calculate N based on levels
    P = P_start
    N = P * (m + 1) - m
    for i = 2:level
        P = N
        N = P * (m + 1) - m
    end
    
    A_list, B_list, x_list, x, d_list = generate_data(N, n)
    A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu = to_gpu(A_list, B_list, x_list, x, d_list)

    # Warmup factorization
    data = initialize(P_start * (m + 1) - m, m, n, P_start, A_list, B_list, level)
    data_gpu = initialize(P_start * (m + 1) - m, m, n, P_start, A_list_gpu, B_list_gpu, level)
    cpu_factorize_time = @elapsed factorize!(data)
    cpu_solve_time = @elapsed solve!(data, d_list, x)
    gpu_factorize_time = @elapsed factorize!(data_gpu)
    gpu_solve_time = @elapsed solve!(data_gpu, d_list_gpu, x_gpu)
    
    GC.gc()
    CUDA.reclaim()

    return cpu_factorize_time, cpu_solve_time, gpu_factorize_time, gpu_solve_time, N
end

function benchmark_block_cholesky(num_trials::Int=10)

    println("Starting warmup...")
    cpu_factorize_time, cpu_solve_time, gpu_factorize_time, gpu_solve_time = run(2, 10, 3, 3)

    println("Starting Block Cholesky Benchmark Suite")
    println("=======================================")

    # Test cases with different parameters
    test_cases = [
        (2, 100, 3, 3),    # Small problem
        (4, 100, 3, 4),    # Medium problem
        (50, 100, 3, 2),
        (8, 100, 6, 4),    # Large problem
    ]

    results = []

    for (m, n, P_start, level) in test_cases
        P = P_start
        N = P * (m + 1) - m
        for i = 2:level
            P = N
            N = P * (m + 1) - m
        end
        cpu_factorize_times = Float64[]
        cpu_solve_times = Float64[]
        gpu_factorize_times = Float64[]
        gpu_solve_times = Float64[]

        for _ in tqdm(1:num_trials)
            cpu_factorize_time, cpu_solve_time, gpu_factorize_time, gpu_solve_time = run(m, n, P_start, level)
            push!(cpu_factorize_times, cpu_factorize_time)
            push!(cpu_solve_times, cpu_solve_time)
            push!(gpu_factorize_times, gpu_factorize_time)
            push!(gpu_solve_times, gpu_solve_time)
        end

        # Calculate statistics
        cpu_factorize_mean = mean(cpu_factorize_times)
        cpu_factorize_std = std(cpu_factorize_times)
        cpu_solve_mean = mean(cpu_solve_times)
        cpu_solve_std = std(cpu_solve_times)
        gpu_factorize_mean = mean(gpu_factorize_times)
        gpu_factorize_std = std(gpu_factorize_times)
        gpu_solve_mean = mean(gpu_solve_times)
        gpu_solve_std = std(gpu_solve_times)
        speedup_factorize = cpu_factorize_mean / gpu_factorize_mean
        speedup_solve = cpu_solve_mean / gpu_solve_mean

        # Store results
        push!(results, (m, n, N, level, cpu_factorize_mean, cpu_factorize_std, cpu_solve_mean, cpu_solve_std, gpu_factorize_mean, gpu_factorize_std, gpu_solve_mean, gpu_solve_std, speedup_factorize, speedup_solve))
    end

    # Print summary table
    println("\nBenchmark Summary")
    println("=================")
    println("m  n    N    Level    CPU F(s)     CPU S(s)     GPU F(s)    GPU S(s)    Speedup(F)  Speedup(S)")
    println("----------------------------------------------------------------------------------------------------------")
    for r in results
        @printf("%d  %d  %d  %d  %.4f ± %.4f  %.4f ± %.4f  %.4f ± %.4f  %.4f ± %.4f  %.2fx  %.2fx\n",
                 r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14])
    end
end

# Call the benchmark function
benchmark_block_cholesky(10)