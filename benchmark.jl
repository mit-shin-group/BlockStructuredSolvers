using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse

using LDLFactorizations
using HSL

import Pkg
include("TriDiagBlockNestedv2.jl")
include("benchmark_utils.jl")

using Statistics, Printf

######

n = 50 # size of each block
m = 2 # number of blocks between separators
N = 55 # number of diagonal blocks
P = Int((N + m) / (m+1)) # number of separators
level = 3; # number of nested level


# for i = 1:1000

#     BigMatrix, d, x_true, A_list, B_list = generate_tridiagonal_system(N, n);

#     #################################################

#     BigMatrix_57 = Ma57(BigMatrix);

#     @time ma57_factorize!(BigMatrix_57);
#     @time x = ma57_solve(BigMatrix_57, d);

#     norm(x - x_true)

#     #################################################

#     BigMatrix_sparse = SparseMatrixCSC(BigMatrix);

#     @time F = cholesky(BigMatrix_sparse);

#     @time x = F \ d;

#     norm(x - x_true)

#     #################################################

#     @time LDLT = ldl(BigMatrix);  # LDLᵀ factorization of A

#     @time x = LDLT \ d;  # solves Ax = b

#     norm(x - x_true)

#     #*********************
#     data = initialize(N, m, n, P, A_list, B_list, level);

#     @time factorize(data);

#     x = zeros(data.N * n);

#     @time solve(data, d, x)

#     norm(x - x_true)

# end

function benchmark_factorization_and_solve(N, n, m, P, level, iter)
    # Storage for times
    ma57_factor_times = Float64[]
    ma57_solve_times = Float64[]

    chol_factor_times = Float64[]
    chol_solve_times = Float64[]

    ldl_factor_times = Float64[]
    ldl_solve_times = Float64[]

    custom_factor_times = Float64[]
    custom_solve_times = Float64[]

    for i = 1:iter
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
        # **Method 4: Custom Solver**
        #################################################
        data = initialize(N, m, n, P, A_list, B_list, level)

        custom_factor_time = @elapsed factorize(data)

        x = zeros(data.N * n)
        custom_solve_time = @elapsed solve(data, d, x)

        push!(custom_factor_times, custom_factor_time)
        push!(custom_solve_times, custom_solve_time)
    end

    # Compute and print the average times
    println("Average Factorization and Solve Times over ", iter, " Runs:")
    println("---------------------------------------------------")
    @printf("MA57 - Factorize: %.6fs, Solve: %.6fs\n", mean(ma57_factor_times), mean(ma57_solve_times))
    @printf("Cholesky - Factorize: %.6fs, Solve: %.6fs\n", mean(chol_factor_times), mean(chol_solve_times))
    @printf("LDLᵀ - Factorize: %.6fs, Solve: %.6fs\n", mean(ldl_factor_times), mean(ldl_solve_times))
    @printf("Custom - Factorize: %.6fs, Solve: %.6fs\n", mean(custom_factor_times), mean(custom_solve_times))
end


benchmark_factorization_and_solve(N, n, m, P, level, 100)