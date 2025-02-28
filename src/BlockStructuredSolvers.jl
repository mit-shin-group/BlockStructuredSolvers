module BlockStructuredSolvers

using LinearAlgebra

export BlockStructuredData, initialize, factorize!, solve!

mutable struct BlockStructuredData{ #TODO create initialize function
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

    M_n_1::MS
    M_n_2::MS
    M_2n::MS
    M_n_2n_1::MS
    M_n_2n_2::MS
    M_mn_2n_1::MS
    M_mn_2n_2::MS
    U_n::UpperTriangular{T, MS}
    U_mn::UpperTriangular{T, MS}

    v_n_1::MU
    v_n_2::MU

    NextData::Union{BlockStructuredData, Nothing}

    next_idx::Vector{Int}
    next_x::MU

end

function initialize(N, m, n, P, A_list, B_list, level)

    I_separator = 1:(m+1):N

    LHS_A_list = zeros(P, n, n);
    LHS_B_list = zeros(P-1, n, n);

    MA_list = zeros(P-1, m*n, m*n);

    RHS = zeros(P * n);

    MA_chol_A_list = zeros(P-1, m, n, n);
    MA_chol_B_list = zeros(P-1, m-1, n, n);

    LHS_chol_A_list = zeros(P, n, n);
    LHS_chol_B_list = zeros(P-1, n, n);

    factor_list = zeros(P-1, m*n, 2*n);

    M_n_1 = similar(A_list, n, n);
    M_n_2 = similar(A_list, n, n);
    U_mn = UpperTriangular(zeros(m*n, m*n));

    M_2n = similar(A_list, 2*n, 2*n);
    M_n_2n_1 = zeros(n, 2*n);
    M_n_2n_2 = zeros(n, 2*n);
    M_mn_2n_1 = zeros(m*n, 2*n);
    M_mn_2n_2 = zeros(m*n, 2*n);

    U_n = UpperTriangular(zeros(n, n));

    v_n_1 = zeros(n);
    v_n_2 = zeros(n);

    next_idx = Int[]

    for j = I_separator

        append!(next_idx, (j-1)*n+1:j*n)
        
    end

    next_x = zeros(P*n);

    data = BlockStructuredData(
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
        M_n_1,
        M_n_2,
        M_2n,
        M_n_2n_1,
        M_n_2n_2,
        M_mn_2n_1,
        M_mn_2n_2,
        U_n,
        U_mn,
        v_n_1,
        v_n_2,
        nothing,
        next_idx,
        next_x
    );

    prev_data = data;

    for i = 2:level

        N = P;
        m = 2;
        P = Int((N + m) / (m+1));

        I_separator = 1:(m+1):N

        LHS_A_list = zeros(P, n, n);
        LHS_B_list = zeros(P-1, n, n);

        RHS = zeros(P * n);

        MA_chol_A_list = zeros(P-1, m, n, n);
        MA_chol_B_list = zeros(P-1, m-1, n, n);
    
        LHS_chol_A_list = zeros(P, n, n);
        LHS_chol_B_list = zeros(P-1, n, n);

        factor_list = zeros(P-1, m*n, 2*n);

        M_n_1 = similar(A_list, n, n);
        M_n_2 = similar(A_list, n, n);
        U_mn = UpperTriangular(zeros(m*n, m*n));

        M_2n = similar(A_list, 2*n, 2*n);
        M_n_2n_1 = zeros(n, 2*n);
        M_n_2n_2 = zeros(n, 2*n);
        M_mn_2n_1 = zeros(m*n, 2*n);
        M_mn_2n_2 = zeros(m*n, 2*n);

        U_n = UpperTriangular(zeros(n, n));

        v_n_1 = zeros(n);
        v_n_2 = zeros(n);

        next_idx = Int[]

        for j = I_separator

            append!(next_idx, (j-1)*n+1:j*n)
            
        end

        next_x = zeros(P*n);

        next_data = BlockStructuredData(
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
            M_n_1,
            M_n_2,
            M_2n,
            M_n_2n_1,
            M_n_2n_2,
            M_mn_2n_1,
            M_mn_2n_2,
            U_n,
            U_mn,
            v_n_1,
            v_n_2,
            nothing,
            next_idx,
            next_x,
            );

        prev_data.NextData = next_data;
        prev_data = next_data;
    end

    return data

end


function cholesky_factorize(A_list, B_list, M_chol_A_list, M_chol_B_list, A, B, U, N, n)

    copyto!(A, view(A_list, 1, :, :))
    cholesky!(Hermitian(A))
    copyto!(U, UpperTriangular(A))
    copyto!(view(M_chol_A_list, 1, :, :), U.data)

    # Iterate over remaining blocks
    for i = 2:N

        # Solve for L_{i, i-1}
        copyto!(A, view(B_list, i-1, :, :))
        ldiv!(B, U', A)
        copyto!(view(M_chol_B_list, i-1, :, :), B)

        copyto!(A, view(A_list, i, :, :))

        # Compute Schur complement
        # mul!(A, B', B, -1.0, 1.0)
        BLAS.gemm!('T', 'N', -1.0, B, B, 1.0, A)
        
        # Compute Cholesky factor for current block
        cholesky!(Hermitian(A));
        copyto!(U, UpperTriangular(A));
        copyto!(view(M_chol_A_list, i, :, :), U.data)

    end

end

function cholesky_solve(M_chol_A_list, M_chol_B_list, d, A, u, v, N, n)
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n)

    LAPACK.trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        BLAS.gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

    LAPACK.trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        BLAS.gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

end

function cholesky_solve_matrix(M_chol_A_list, M_chol_B_list, d, A, u, v, N, n) #TODO merge two solves
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n, :)

    LAPACK.trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n, :) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        BLAS.gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

    LAPACK.trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n, :) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        BLAS.gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

end

function factorize!(
    data::BlockStructuredData
)

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

A = data.M_n_1
B = data.M_n_2
M_2n = data.M_2n
M_mn_2n_1 = data.M_mn_2n_1
M_mn_2n_2 = data.M_mn_2n_2

U = data.U_n
u = data.M_n_2n_1
v = data.M_n_2n_2

@views for i = 1:P-1 #TODO get rid of views

    # Compute inverse of block tridiagonal matrices to compute inverse of MA (top left of Schur complement)
    cholesky_factorize(
        A_list[I_separator[i]+1:I_separator[i]+m, :, :], 
        B_list[I_separator[i]+1:I_separator[i]+m-1, :, :], 
        MA_chol_A_list[i, :, :, :],
        MA_chol_B_list[i, :, :, :],
        A,
        B,
        U,
        m, 
        n)

    # MA_chol_list[i, :, :] .= MA_chol
    
    M_mn_2n_1[1:n, 1:n] .= B_list[I_separator[i], :, :]'
    M_mn_2n_1[m*n-n+1:m*n, n+1:2*n] .= B_list[I_separator[i+1]-1, :, :]

    M_mn_2n_2 .= M_mn_2n_1

    # ldiv!(MA_chol', M_mn_2n_2)
    # ldiv!(MA_chol, M_mn_2n_2)

    cholesky_solve_matrix(MA_chol_A_list[i, :, :, :], MA_chol_B_list[i, :, :, :], M_mn_2n_2, A, u, v, m, n)

    factor_list[i, :, :] = M_mn_2n_2

    mul!(M_2n, M_mn_2n_1', M_mn_2n_2)

    LHS_A_list[i, :, :] .-= M_2n[1:n, 1:n]
    LHS_A_list[i+1, :, :] .-= M_2n[n+1:2*n, n+1:2*n]
    LHS_B_list[i, :, :] .-= M_2n[1:n, n+1:2*n]

    LHS_A_list[i, :, :] .+= A_list[I_separator[i], :, :]

end

copyto!(A, view(LHS_A_list, P, :, :))
A .+= view(A_list, I_separator[P], :, :) #TODO how to get rid of .+=
copyto!(view(LHS_A_list, P, :, :), A)

if isnothing(data.NextData)

    LHS_chol_A_list = data.LHS_chol_A_list
    LHS_chol_B_list = data.LHS_chol_B_list

    cholesky_factorize(LHS_A_list, LHS_B_list, LHS_chol_A_list, LHS_chol_B_list, A, B, U, P, n)

else

    data.NextData.A_list = LHS_A_list
    data.NextData.B_list = LHS_B_list
    factorize!(data.NextData)

end

end

function solve!(data::BlockStructuredData, d, x)

    P = data.P
    n = data.n
    m = data.m

    I_separator = data.I_separator
    B_list = data.B_list
    MA_chol_A_list = data.MA_chol_A_list
    MA_chol_B_list = data.MA_chol_B_list
    factor_list = data.factor_list

    B = data.M_n_2
    U = data.U_mn
    M_mn_2n_1 = data.M_mn_2n_1

    RHS = data.RHS
    u = data.v_n_1
    v = data.v_n_2

    # Assign RHS from d
    @inbounds for j = 1:P
        view(RHS, (j-1)*n+1:j*n) .= view(d, I_separator[j]*n-n+1:I_separator[j]*n)
    end

    # Compute RHS from Schur complement
    @inbounds for i = 1:P-1
        M_mn_2n_1 .= view(factor_list, i, :, :)
        mul!(view(RHS, (i-1)*n+1:(i+1)*n), M_mn_2n_1', view(d, I_separator[i]*n+1:I_separator[i+1]*n-n), -1.0, 1.0)
    end

    # Solve system
    if isnothing(data.NextData)

        LHS_chol_A_list = data.LHS_chol_A_list
        LHS_chol_B_list = data.LHS_chol_B_list

        cholesky_solve(LHS_chol_A_list, LHS_chol_B_list, RHS, B, u, v, P, n)

        # Assign RHS to x for separators
        @inbounds for i = 1:P
            view(x, I_separator[i]*n-n+1:I_separator[i]*n) .= view(RHS, (i-1)*n+1:i*n)
        end
    else
        data.next_x .= view(x, data.next_idx)
        solve!(data.NextData, RHS, data.next_x)
        view(x, data.next_idx) .= data.next_x
    end

    # Update d after Schur solve
    @inbounds for j = 1:P-1
        B .= view(B_list, I_separator[j], :, :)
        BLAS.gemm!('T', 'N', -1.0, B, view(x, I_separator[j]*n-n+1:I_separator[j]*n), 1.0, view(d, I_separator[j]*n+1:I_separator[j]*n+n))

        B .= view(B_list, I_separator[j+1]-1, :, :)
        mul!(view(d, I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n), B, view(x, I_separator[j+1]*n-n+1:I_separator[j+1]*n), -1.0, 1.0)
    end

    # Solve for non-separators
    @inbounds for i = 1:P-1

        cholesky_solve(view(MA_chol_A_list, i, :, :, :), view(MA_chol_B_list, i, :, :, :), view(d, I_separator[i]*n+1:I_separator[i+1]*n-n), B, u, v, m, n)
        view(x, I_separator[i]*n+1:I_separator[i+1]*n-n) .= view(d, I_separator[i]*n+1:I_separator[i+1]*n-n)
    end

    return nothing
end


end