using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse

using LDLFactorizations
using HSL

import Pkg
include("TriDiagBlockNestedv2.jl")
import .TriDiagBlockNested: TriDiagBlockDataNested, initialize, factorize, solve

include("benchmark_utils.jl")

using Statistics, Printf, ProgressBars


######

n = 10 # size of each block
m = 5 # number of blocks between separators
N = 685 # number of diagonal blocks
P = Int((N + m) / (m+1)) # number of separators
level = 2; # number of nested level

function benchmark_factorization_and_solve(N, n, m, P, level, iter)
    # Storage for times
    ma57_factor_times = Float64[]
    ma57_solve_times = Float64[]

    chol_factor_times = Float64[]
    chol_solve_times = Float64[]

    ldl_factor_times = Float64[]
    ldl_solve_times = Float64[]

    our_factor_times = Float64[]
    our_solve_times = Float64[]

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

        chol_factor_time = @elapsed F = cholesky(BigMatrix_sparse) #TODO, pre allocate F
        chol_solve_time = @elapsed x = F \ d

        push!(chol_factor_times, chol_factor_time)
        push!(chol_solve_times, chol_solve_time)

        #################################################
        # **Method 3: LDLᵀ Factorization**
        #################################################
        ldl_factor_time = @elapsed LDLT = ldl(BigMatrix) # pre allocate LDLT, separate factorize
        ldl_solve_time = @elapsed x = LDLT \ d

        push!(ldl_factor_times, ldl_factor_time)
        push!(ldl_solve_times, ldl_solve_time)

        #################################################
        # **Method 4: Our Solver**
        #################################################
        data = initialize(N, m, n, P, A_list, B_list, level)

        our_factor_time = @elapsed factorize(data)

        x = zeros(data.N * n)
        our_solve_time = @elapsed solve(data, d, x)

        push!(our_factor_times, our_factor_time)
        push!(our_solve_times, our_solve_time)
    end

    # Compute and print the average times
    println("Average Factorization and Solve Times over ", iter, " Runs:")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ma57_factor_times) * 1000, mean(ma57_solve_times) * 1000)
    @printf("Cholesky - Factorize: %.6f ms, Solve: %.6f ms\n", mean(chol_factor_times) * 1000, mean(chol_solve_times) * 1000)
    @printf("LDLᵀ - Factorize: %.6f ms, Solve: %.6f ms\n", mean(ldl_factor_times) * 1000, mean(ldl_solve_times) * 1000)
    @printf("Ours - Factorize: %.6f ms, Solve: %.6f ms\n", mean(our_factor_times) * 1000, mean(our_solve_times) * 1000)
end


benchmark_factorization_and_solve(N, n, m, P, level, 100)