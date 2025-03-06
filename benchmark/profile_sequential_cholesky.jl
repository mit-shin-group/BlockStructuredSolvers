using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse
using Printf, ProgressBars, Statistics
using Random
using TimerOutputs

using LinearAlgebra.LAPACK: potrf!
using LinearAlgebra.BLAS: gemm!, gemv!, trsv!, trsm! #TODO check version of BLAS

import Pkg
include("utils.jl")

# Problem parameters
n = 100  # size of each block
N = 55   # number of diagonal blocks
iter = 3 # number of iterations for profiling
seed = 42 # random seed for reproducibility

    const to_factor = TimerOutput()
    const to_solve = TimerOutput()

struct BlockTriDiagData_sequential_cholesky_factor{
    T, 
    MT <: AbstractArray{T, 3}}

    N::Int
    n::Int

    A_list::MT
    B_list::MT

end

function initialize_sequential_cholesky_factor(N, n, A_list, B_list)

    data = BlockTriDiagData_sequential_cholesky_factor(
        N,
        n,
        A_list,
        B_list
    )

    return data

end


function cholesky_factorize!(A_list, B_list, N)

    @timeit to_factor "potrf" begin potrf!('U', view(A_list, :, :, 1)) end

    for i = 2:N

        @timeit to_factor "trsm" begin trsm!('L', 'U', 'T', 'N', 1.0, view(A_list, :, :, i-1), view(B_list, :, :, i-1)) end
        @timeit to_factor "gemm" begin gemm!('T', 'N', -1.0, view(B_list, :, :, i-1), view(B_list, :, :, i-1), 1.0, view(A_list, :, :, i)) end
        @timeit to_factor "potrf" begin potrf!('U', view(A_list, :, :, i)) end

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 1}}

    @timeit to_solve "trsv" begin trsv!('U', 'T', 'N', view(M_chol_A_list, :, :, 1), view(d, 1:n)) end

    for i = 2:N #TODO pprof.jl

        @timeit to_solve "gemv" begin gemv!('T', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n), 1.0, view(d, (i-1)*n+1:i*n)) end

        @timeit to_solve "trsv" begin trsv!('U', 'T', 'N',  view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n)) end

    end

    @timeit to_solve "trsv" begin trsv!('U', 'N', 'N', view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n)) end

    for i = N-1:-1:1

        @timeit to_solve "gemv" begin gemv!('N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n), 1.0, view(d, (i-1)*n+1:i*n)) end

        @timeit to_solve "trsv" begin trsv!('U', 'N', 'N', view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n)) end

    end

end


function factorize_sequential_cholesky_factor!(
    data::BlockTriDiagData_sequential_cholesky_factor
)

N = data.N

A_list = data.A_list
B_list = data.B_list

cholesky_factorize!(A_list, B_list, N)

end

function solve_sequential_cholesky_factor!(data::BlockTriDiagData_sequential_cholesky_factor, d, x)

    N = data.N
    n = data.n
    A_list = data.A_list
    B_list = data.B_list

    cholesky_solve!(A_list, B_list, d, N, n)

    x .= d

    return nothing
end

function profile_sequential_cholesky(n, N, iter, seed)
    # Set random seed
    Random.seed!(seed)
    
    println("Profiling Sequential Cholesky Solver")
    println("=====================================")
    println("Block size: $n")
    println("Number of blocks: $N")
    println("Number of iterations: $iter")
    println("Random seed: $seed")
    println("\n")

    # Create separate TimerOutputs for factorization and solve

    for run in 1:iter
        println("Run $run/$iter")
        
        _, d, x_true, A_list, B_list = generate_tridiagonal_system(N, n)

        # Initialize solver
        data = initialize_sequential_cholesky_factor(N, n, A_list, B_list)
        x = zeros(data.N * n)

        # Profile factorization with TimerOutputs
        println("\nProfiling Factorization:")
        @timeit to_factor "Factorization" begin
            factorize_sequential_cholesky_factor!(data)
        end

        # Profile solve with TimerOutputs
        println("\nProfiling Solve:")
        @timeit to_solve "Solve" begin
            solve_sequential_cholesky_factor!(data, d, x)
        end

        println("----------------------------------------")
    end

    println("\nFactorization TimerOutputs summary:")
    println("=====================================")
    show(to_factor)
    println("\n\nSolve TimerOutputs summary:")
    println("============================")
    show(to_solve)
end

# Run the profiling
profile_sequential_cholesky(n, N, iter, seed)