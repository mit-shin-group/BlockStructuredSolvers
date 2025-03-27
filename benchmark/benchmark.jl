using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse
using CUDA
using Random
using LDLFactorizations
using HSL
using CUDSS
using MKL
using BlockStructuredSolvers

import Pkg
include("utils.jl")

using Printf, ProgressBars, Statistics


######

println(LinearAlgebra.BLAS.get_config())

n = 32 # size of each block
m = 2 # number of blocks between separators
P_last = 3 # number of separators
level = 4; # number of nested level
N_last = P_last * (m + 1) - m; # number of diagonal blocks
seed = 42 # random seed for reproducibility

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
    BigMatrix, d = construct_block_tridiagonal(A_list, B_list, d_list)

    BigMatrix_57 = Ma57(BigMatrix)
    d_57 = deepcopy(d)
    ma57_factorize_time = @elapsed ma57_factorize!(BigMatrix_57)
    ma57_solve_time = @elapsed ma57_solve(BigMatrix_57, d_57)

    BigMatrix_57 = nothing

    # CHOLMOD
    chol_factor_time = @elapsed F = cholesky(BigMatrix)
    chol_solve_time = @elapsed F \ d

    F = nothing

    # LDLFactorizations 
    ldl_factor_time = @elapsed LDLT = ldl(BigMatrix; P=Vector(1:N*n))
    ldl_solve_time = @elapsed LDLT \ d

    LDLT = nothing

    # CUDSS
    BigMatrix_cudss = CUSPARSE.CuSparseMatrixCSR(BigMatrix)
    d_cudss = CuArray(d)
    chol = CUDSS.cholesky(BigMatrix_cudss; check=false)
    cudss_factor_time = CUDA.@elapsed blocking=true CUDSS.cholesky(BigMatrix_cudss; check=false)
    
    # Time CUDSS solve using events
    x_cudss = deepcopy(d_cudss)
    cudss_solve_time = CUDA.@elapsed blocking=true cudss("solve", chol, x_cudss, d_cudss)

    BigMatrix_cudss = nothing
    chol = nothing

    # Warmup factorization
    data = initialize(P_start * (m + 1) - m, m, n, P_start, A_list, B_list, level)
    data_gpu = initialize(P_start * (m + 1) - m, m, n, P_start, A_list_gpu, B_list_gpu, level)
    data_sequential = initialize(N, n, A_list, B_list)
    cpu_factorize_time = @elapsed factorize!(data)
    cpu_solve_time = @elapsed solve!(data, d_list, x)
    gpu_factorize_time = @elapsed CUDA.@sync factorize!(data_gpu)
    gpu_solve_time = @elapsed CUDA.@sync solve!(data_gpu, d_list_gpu, x_gpu)
    sequential_factorize_time = @elapsed factorize!(data_sequential)
    sequential_solve_time = @elapsed solve!(data_sequential, d_list, x)
    
    GC.gc()
    CUDA.reclaim()

    results = [ma57_factorize_time, 
    ma57_solve_time, 
    chol_factor_time, 
    chol_solve_time, 
    ldl_factor_time, 
    ldl_solve_time, 
    cudss_factor_time, 
    cudss_solve_time, 
    cpu_factorize_time, 
    cpu_solve_time, 
    gpu_factorize_time, 
    gpu_solve_time, 
    sequential_factorize_time, 
    sequential_solve_time]
    
    return results
end

function benchmark_factorization_and_solve(iter)

    # Storage for times
    ma57_factor_times = Float64[]
    ma57_solve_times = Float64[]

    chol_factor_times = Float64[]
    chol_solve_times = Float64[]

    ldl_factor_times = Float64[]
    ldl_solve_times = Float64[]

    cudss_factor_times = Float64[]
    cudss_solve_times = Float64[]

    cpu_factor_times = Float64[]
    cpu_solve_times = Float64[]

    gpu_factor_times = Float64[]
    gpu_solve_times = Float64[]

    sequential_factor_times = Float64[]
    sequential_solve_times = Float64[]

    println("Starting warmup...")

    _ = run(2, 10, 3, 1)

    (m, n, P_start, level) = (33, 128, 33, 1) #(4, 200, 3, 4)

    P = P_start
    N = P * (m + 1) - m
    for i = 2:level
        P = N
        N = P * (m + 1) - m
    end

    println("Starting benchmark...")

    for i = tqdm(1:iter)

        results = run(m, n, P_start, level)

        push!(ma57_factor_times, results[1])
        push!(ma57_solve_times, results[2])

        push!(chol_factor_times, results[3])
        push!(chol_solve_times, results[4])

        push!(ldl_factor_times, results[5])
        push!(ldl_solve_times, results[6])

        push!(cudss_factor_times, results[7])
        push!(cudss_solve_times, results[8])

        push!(cpu_factor_times, results[9])
        push!(cpu_solve_times, results[10])

        push!(gpu_factor_times, results[11])
        push!(gpu_solve_times, results[12])

        push!(sequential_factor_times, results[13])
        push!(sequential_solve_times, results[14])
    end

    # Compute and print the average times
    println("\nAverage Factorization and Solve Times over ", iter, " Runs:", " N: $N, n: $n, P: $P, m: $m, level: $level")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ma57_factor_times) * 1000, mean(ma57_solve_times) * 1000)
    @printf("CHOLMOD - Factorize: %.6f ms, Solve: %.6f ms\n", mean(chol_factor_times) * 1000, mean(chol_solve_times) * 1000)
    @printf("LDLFactorizations - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ldl_factor_times) * 1000, mean(ldl_solve_times) * 1000)
    @printf("CUDSS - Factorize: %.6f ms, Solve: %.6f ms\n", mean(cudss_factor_times) * 1000, mean(cudss_solve_times) * 1000)
    @printf("CPU - Factorize: %.6f ms, Solve: %.6f ms\n", mean(cpu_factor_times) * 1000, mean(cpu_solve_times) * 1000)
    @printf("GPU - Factorize: %.6f ms, Solve: %.6f ms\n", mean(gpu_factor_times) * 1000, mean(gpu_solve_times) * 1000)
    @printf("Sequential - Factorize: %.6f ms, Solve: %.6f ms\n", mean(sequential_factor_times) * 1000, mean(sequential_solve_times) * 1000)
    # Also print standard deviations
    println("\nStandard Deviations:")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6f ms, Solve: %.6f ms\n", std(ma57_factor_times) * 1000, std(ma57_solve_times) * 1000)
    @printf("CHOLMOD - Factorize: %.6f ms, Solve: %.6f ms\n", std(chol_factor_times) * 1000, std(chol_solve_times) * 1000)
    @printf("LDLFactorizations - Factorize: %.6f ms, Solve: %.6f ms\n", std(ldl_factor_times) * 1000, std(ldl_solve_times) * 1000)
    @printf("CUDSS - Factorize: %.6f ms, Solve: %.6f ms\n", std(cudss_factor_times) * 1000, std(cudss_solve_times) * 1000)
    @printf("CPU - Factorize: %.6f ms, Solve: %.6f ms\n", std(cpu_factor_times) * 1000, std(cpu_solve_times) * 1000)
    @printf("GPU - Factorize: %.6f ms, Solve: %.6f ms\n", std(gpu_factor_times) * 1000, std(gpu_solve_times) * 1000)
    @printf("Sequential - Factorize: %.6f ms, Solve: %.6f ms\n", std(sequential_factor_times) * 1000, std(sequential_solve_times) * 1000)
end

# Run benchmark with warmup
benchmark_factorization_and_solve(10)