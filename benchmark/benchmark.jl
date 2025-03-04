using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse

using LDLFactorizations
using HSL
using BlockStructuredSolvers

import Pkg
include("utils.jl")

using Printf, ProgressBars, Statistics


######

n = 10 # size of each block
m = 2 # number of blocks between separators
P_last = 3 # number of separators
level = 3; # number of nested level
N_last = P_last * (m + 1) - m; # number of diagonal blocks

function benchmark_factorization_and_solve(N_last, n, m, P_last, level, iter)

    N = N_last;
    P = P_last;

    for i = 2:level
        P = N;
        N = P * (m + 1) - m;
    end

    # Storage for times
    ma57_factor_times = Float64[]
    ma57_solve_times = Float64[]

    chol_factor_times = Float64[]
    chol_solve_times = Float64[]

    ldl_factor_times = Float64[]
    ldl_solve_times = Float64[]

    our_factor_times = Float64[]
    our_solve_times = Float64[]

    our_factor_times_2 = Float64[]
    our_solve_times_2 = Float64[]

    our_factor_times_3 = Float64[]
    our_solve_times_3 = Float64[]

    # Warmup run to trigger compilation
    println("Performing warmup runs...")
    BigMatrix_warmup, d_warmup, x_true_warmup, A_list_warmup, B_list_warmup = generate_tridiagonal_system(N, n)
    
    # Warmup MA57
    BigMatrix_57_warmup = Ma57(BigMatrix_warmup)
    ma57_factorize!(BigMatrix_57_warmup)
    ma57_solve(BigMatrix_57_warmup, d_warmup)
    
    # Warmup Cholesky
    BigMatrix_sparse_warmup = SparseMatrixCSC(BigMatrix_warmup)
    F_warmup = cholesky(BigMatrix_sparse_warmup)
    F_warmup \ d_warmup
    
    # Warmup LDLᵀ
    LDLT_warmup = ldl(BigMatrix_warmup)
    LDLT_warmup \ d_warmup
    
    # Warmup our solvers
    data_warmup = initialize(N_last, m, n, P_last, A_list_warmup, B_list_warmup, level)
    factorize!(data_warmup)
    x_warmup = zeros(data_warmup.N * n)
    solve!(data_warmup, d_warmup, x_warmup)
    
    data_warmup_2 = initialize_full_cholesky_factor(N, m, n, P, A_list_warmup, B_list_warmup, level)
    factorize_full_cholesky_factor!(data_warmup_2)
    x_warmup_2 = zeros(data_warmup_2.N * n)
    solve_full_cholesky_factor!(data_warmup_2, d_warmup, x_warmup_2)

    data_warmup_3 = initialize_sequential_cholesky_factor(N, n, A_list_warmup, B_list_warmup)
    factorize_sequential_cholesky_factor!(data_warmup_3)
    x_warmup_3 = zeros(data_warmup_3.N * n)
    solve_sequential_cholesky_factor!(data_warmup_3, d_warmup, x_warmup_3)
    
    println("Starting actual benchmark...")
    
    for i = tqdm(1:iter)
        # Generate problem instance
        BigMatrix, d, x_true, A_list, B_list = generate_tridiagonal_system(N, n)

        #################################################
        # **Method 1: MA57 Solver**
        #################################################
        BigMatrix_57 = Ma57(BigMatrix)

        ma57_factor_time = @elapsed ma57_factorize!(BigMatrix_57)
        ma57_solve_time = @elapsed x = ma57_solve(BigMatrix_57, d)

        push!(ma57_factor_times, ma57_factor_time)
        push!(ma57_solve_times, ma57_solve_time)

        #################################################
        # **Method 2: Cholesky Factorization**
        #################################################
        BigMatrix_sparse = SparseMatrixCSC(BigMatrix)

        chol_factor_time = @elapsed F = cholesky(BigMatrix_sparse)
        chol_solve_time = @elapsed x = F \ d

        push!(chol_factor_times, chol_factor_time)
        push!(chol_solve_times, chol_solve_time)

        #################################################
        # **Method 3: LDLᵀ Factorization**
        #################################################
        ldl_factor_time = @elapsed LDLT = ldl(BigMatrix)
        ldl_solve_time = @elapsed x = LDLT \ d

        push!(ldl_factor_times, ldl_factor_time)
        push!(ldl_solve_times, ldl_solve_time)

        #################################################
        # **Method 4: Our Solver**
        #################################################
        data = initialize(N_last, m, n, P_last, A_list, B_list, level)

        our_factor_time = @elapsed factorize!(data)

        x = zeros(data.N * n)
        our_solve_time = @elapsed solve!(data, d, x)

        push!(our_factor_times, our_factor_time)
        push!(our_solve_times, our_solve_time)

        #################################################
        # **Method 5: Our Solver (Full Cholesky)**
        #################################################
        data2 = initialize_full_cholesky_factor(N, m, n, P, A_list, B_list, level)

        our_factor_time_2 = @elapsed factorize_full_cholesky_factor!(data2)

        x2 = zeros(data2.N * n)
        our_solve_time_2 = @elapsed solve_full_cholesky_factor!(data2, d, x2)

        push!(our_factor_times_2, our_factor_time_2)
        push!(our_solve_times_2, our_solve_time_2)

        #################################################
        # **Method 6: Sequential Cholesky Factor**
        #################################################
        data3 = initialize_sequential_cholesky_factor(N, n, A_list, B_list)

        our_factor_time_3 = @elapsed factorize_sequential_cholesky_factor!(data3)

        x3 = zeros(data3.N * n)
        our_solve_time_3 = @elapsed solve_sequential_cholesky_factor!(data3, d, x3)

        push!(our_factor_times_3, our_factor_time_3)
        push!(our_solve_times_3, our_solve_time_3)
    end

    # Compute and print the average times
    println("\nAverage Factorization and Solve Times over ", iter, " Runs:")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ma57_factor_times) * 1000, mean(ma57_solve_times) * 1000)
    @printf("Cholesky - Factorize: %.6f ms, Solve: %.6f ms\n", mean(chol_factor_times) * 1000, mean(chol_solve_times) * 1000)
    @printf("LDLᵀ - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ldl_factor_times) * 1000, mean(ldl_solve_times) * 1000)
    @printf("Ours - Factorize: %.6f ms, Solve: %.6f ms\n", mean(our_factor_times) * 1000, mean(our_solve_times) * 1000)
    @printf("Ours (Full) - Factorize: %.6f ms, Solve: %.6f ms\n", mean(our_factor_times_2) * 1000, mean(our_solve_times_2) * 1000)
    @printf("Ours (Sequential) - Factorize: %.6f ms, Solve: %.6f ms\n", mean(our_factor_times_3) * 1000, mean(our_solve_times_3) * 1000)
    # Also print standard deviations
    println("\nStandard Deviations:")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6f ms, Solve: %.6f ms\n", std(ma57_factor_times) * 1000, std(ma57_solve_times) * 1000)
    @printf("Cholesky - Factorize: %.6f ms, Solve: %.6f ms\n", std(chol_factor_times) * 1000, std(chol_solve_times) * 1000)
    @printf("LDLᵀ - Factorize: %.6f ms, Solve: %.6f ms\n", std(ldl_factor_times) * 1000, std(ldl_solve_times) * 1000)
    @printf("Ours - Factorize: %.6f ms, Solve: %.6f ms\n", std(our_factor_times) * 1000, std(our_solve_times) * 1000)
    @printf("Ours (Full) - Factorize: %.6f ms, Solve: %.6f ms\n", std(our_factor_times_2) * 1000, std(our_solve_times_2) * 1000)
    @printf("Ours (Sequential) - Factorize: %.6f ms, Solve: %.6f ms\n", std(our_factor_times_3) * 1000, std(our_solve_times_3) * 1000)
end

# Run benchmark with warmup
benchmark_factorization_and_solve(N_last, n, m, P_last, level, 100)