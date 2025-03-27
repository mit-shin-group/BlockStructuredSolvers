using LinearAlgebra, SparseArrays, BlockArrays, SuiteSparse
using Printf, ProgressBars, Statistics
using Random
using TimerOutputs

using LinearAlgebra.LAPACK: potrf!
using LinearAlgebra.BLAS: gemm!, gemv!, trsv!, trsm! #TODO check version of BLAS

import Pkg
include("utils.jl")

# Problem parameters
n = 10 # size of each block
m = 2 # number of blocks between separators
P_last = 3 # number of separators
level = 3; # number of nested level
N_last = P_last * (m + 1) - m; # number of diagonal blocks
seed = 42 # random seed for reproducibility
iter = 3 # number of iterations for profiling

const to_solve = TimerOutput()

function cholesky_factorize!(A_list, B_list, N)

    potrf!('U', view(A_list, :, :, 1))

    for i = 2:N

        trsm!('L', 'U', 'T', 'N', 1.0, view(A_list, :, :, i-1), view(B_list, :, :, i-1))
        gemm!('T', 'N', -1.0, view(B_list, :, :, i-1), view(B_list, :, :, i-1), 1.0, view(A_list, :, :, i))
        potrf!('U', view(A_list, :, :, i))

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

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 2}}

    trsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, 1), view(d, 1:n, :));

    for i = 2:N

        gemm!('T', 'N', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        trsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

    trsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n, :));

    for i = N-1:-1:1

        gemm!('N', 'N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        trsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

end

struct BlockTriDiagData{ #TODO create initialize function
    T, 
    MR <: AbstractArray{T, 4},
    MT <: AbstractArray{T, 3},
    MS <: AbstractArray{T, 2},
    MU <: AbstractArray{T, 1}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    I_separator::StepRange{Int64, Int64}

    A_list::MT
    B_list::MT

    LHS_A_list::MT
    LHS_B_list::MT

    factor_list::MT

    RHS::MU

    MA_chol_A_list::MR
    MA_chol_B_list::MR
    LHS_chol_A_list::MT
    LHS_chol_B_list::MT

    M_2n::MS
    M_mn_2n_1::MS
    M_mn_2n_2::MS

    NextData::Union{BlockTriDiagData, Nothing}

    next_idx::Vector{Int}
    next_x::MU

end

function initialize(N, m, n, P, A_list_final, B_list_final, level)

    data = nothing;

    for i = 1:level

        I_separator = 1:(m+1):N

        LHS_A_list = zeros(n, n, P);
        LHS_B_list = zeros(n, n, P-1);

        RHS = zeros(P * n);

        MA_chol_A_list = zeros(n, n, m, P-1);
        MA_chol_B_list = zeros(n, n, m-1, P-1);
    
        LHS_chol_A_list = zeros(n, n, P);
        LHS_chol_B_list = zeros(n, n, P-1);

        factor_list = zeros(m*n, 2*n, P-1);


        M_2n = zeros(2*n, 2*n);
        M_mn_2n_1 = zeros(m*n, 2*n);
        M_mn_2n_2 = zeros(m*n, 2*n);


        next_idx = Int[]

        for j = I_separator

            append!(next_idx, (j-1)*n+1:j*n)
            
        end

        next_x = zeros(P*n);

        if i == level
            A_list = A_list_final
            B_list = B_list_final
        else
            A_list = zeros(n, n, N);
            B_list = zeros(n, n, N-1);
        end

        data = BlockTriDiagData(
            N, 
            m, 
            n, 
            P, 
            I_separator,
            A_list, 
            B_list,
            LHS_A_list,
            LHS_B_list,
            factor_list,
            RHS,
            MA_chol_A_list,
            MA_chol_B_list,
            LHS_chol_A_list,
            LHS_chol_B_list,
            M_2n,
            M_mn_2n_1,
            M_mn_2n_2,
            data,
            next_idx,
            next_x,
            );

        P = N;
        N = P * (m + 1) - m;

    end

    return data

end

function factorize!(data::BlockTriDiagData)

    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator

    A_list = data.A_list
    B_list = data.B_list

    LHS_A_list = data.LHS_A_list
    LHS_B_list = data.LHS_B_list

    factor_list = data.factor_list

    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list

    M_2n = data.M_2n
    M_mn_2n_1 = data.M_mn_2n_1
    M_mn_2n_2 = data.M_mn_2n_2

    @inbounds for i = 1:P-1
        # Cache views for better performance
        A_block = view(A_list, :, :, I_separator[i]+1:I_separator[i]+m)
        B_block = view(B_list, :, :, I_separator[i]+1:I_separator[i]+m-1)
        MA_chol_A = view(MA_chol_A_list, :, :, :, i)
        MA_chol_B = view(MA_chol_B_list, :, :, :, i)

        MA_chol_A .= A_block
        MA_chol_B .= B_block
        # Compute inverse of block tridiagonal matrices
        cholesky_factorize!(
            MA_chol_A,
            MA_chol_B,
            m
            )
        
        # Cache frequently accessed views
        B_view1 = view(B_list, :, :, I_separator[i])
        B_view2 = view(B_list, :, :, I_separator[i+1]-1)
        
        @views begin
            M_mn_2n_1[1:n, 1:n] .= B_view1'
            M_mn_2n_1[m*n-n+1:m*n, n+1:2*n] .= B_view2
        end

        M_mn_2n_2 .= M_mn_2n_1

        cholesky_solve!(MA_chol_A, MA_chol_B, M_mn_2n_2, m, n)

        factor_list[:, :, i] = M_mn_2n_2

        gemm!('T', 'N', 1.0, M_mn_2n_1, M_mn_2n_2, 0.0, M_2n)

        # Cache views for LHS updates
        lhs_a1 = view(LHS_A_list, :, :, i)
        lhs_a2 = view(LHS_A_list, :, :, i+1)
        lhs_b = view(LHS_B_list, :, :, i)
        
        @views begin
            lhs_a1 .-= M_2n[1:n, 1:n]
            lhs_a2 .-= M_2n[n+1:2*n, n+1:2*n]
            lhs_b .-= M_2n[1:n, n+1:2*n]
        end

        lhs_a1 .+= view(A_list, :, :, I_separator[i])
    end

    # copyto!(A, view(LHS_A_list, :, :, P))
    view(LHS_A_list, :, :, P) .+= view(A_list, :, :, I_separator[P])
    # copyto!(view(LHS_A_list, :, :, P), A)

    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        LHS_chol_A_list .= LHS_A_list
        LHS_chol_B_list .= LHS_B_list

        cholesky_factorize!(LHS_chol_A_list, LHS_chol_B_list, P)
    else
        data.NextData.A_list .= LHS_A_list
        data.NextData.B_list .= LHS_B_list
        factorize!(data.NextData)
    end

end

function solve!(data::BlockTriDiagData, d, x)
    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator
    B_list = data.B_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    factor_list = data.factor_list
    RHS = data.RHS

    # Assign RHS from d
    @inbounds @simd for j = 1:P
        @timeit to_solve "copyto" begin copyto!(view(RHS, (j-1)*n+1:j*n), view(d, I_separator[j]*n-n+1:I_separator[j]*n)) end
    end

    # Compute RHS from Schur complement
    @inbounds for i = 1:P-1
        @timeit to_solve "gemv" begin gemv!('T', -1.0, view(factor_list, :, :, i), view(d, I_separator[i]*n+1:I_separator[i+1]*n-n), 1.0, view(RHS, (i-1)*n+1:(i+1)*n)) end
    end

    # Solve system
    if isnothing(data.NextData)
        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        cholesky_solve!(LHS_chol_A_list, LHS_chol_B_list, RHS, P, n)

        # Assign RHS to x for separators
        @inbounds @simd for i = 1:P
            @timeit to_solve "copyto" begin copyto!(view(x, I_separator[i]*n-n+1:I_separator[i]*n), view(RHS, (i-1)*n+1:i*n)) end
        end
    else
        data.next_x .= view(x, data.next_idx)
        solve!(data.NextData, RHS, data.next_x)
        view(x, data.next_idx) .= data.next_x
    end

    # Update d after Schur solve
    @inbounds for j = 1:P-1
        # Cache views and matrices
        @timeit to_solve "gemv" begin gemv!('T', -1.0, view(B_list, :, :, I_separator[j]), view(x, I_separator[j]*n-n+1:I_separator[j]*n), 1.0, view(d, I_separator[j]*n+1:I_separator[j]*n+n)) end

        @timeit to_solve "gemv" begin gemv!('N', -1.0, view(B_list, :, :, I_separator[j+1]-1), view(x, I_separator[j+1]*n-n+1:I_separator[j+1]*n), 1.0, view(d, I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n)) end
    end

    # Solve for non-separators
    @inbounds for i = 1:P-1
        cholesky_solve!(view(MA_chol_A_list, :, :, :, i), view(MA_chol_B_list, :, :, :, i), view(d, I_separator[i]*n+1:I_separator[i+1]*n-n), m, n)
        @timeit to_solve "copyto" begin copyto!(view(x, I_separator[i]*n+1:I_separator[i+1]*n-n), view(d, I_separator[i]*n+1:I_separator[i+1]*n-n)) end
    end

    return nothing
end


function profile_sequential_cholesky(n, N_last, P_last, iter, level, seed)
    # Set random seed
    Random.seed!(seed)
    
    N = N_last;
    P = P_last;

    for i = 2:level
        P = N;
        N = P * (m + 1) - m;
    end
    
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
        data = initialize(N_last, m, n, P_last, A_list, B_list, level)
        x = zeros(data.N * n)

        # Profile factorization with TimerOutputs
        println("\nProfiling Factorization:")
        factorize!(data)

        # Profile solve with TimerOutputs
        println("\nProfiling Solve:")
        @timeit to_solve "Solve" begin
            solve!(data, d, x)
        end

        println("----------------------------------------")
    end

    println("\n\nSolve TimerOutputs summary:")
    println("============================")
    show(to_solve)
end

# Run the profiling
profile_sequential_cholesky(n, N_last, P_last, iter, level, seed)