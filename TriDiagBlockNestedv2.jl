module TriDiagBlockNested

using LinearAlgebra

export TriDiagBlockDataNested, initialize, factorize, solve

mutable struct TriDiagBlockDataNested{ #TODO create initialize function
    T, 
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

    MA_chol_list::MT
    factor_list::MT

    RHS::MU

    MA_chol::UpperTriangular{T, MS}
    LHS_chol::UpperTriangular{T, MS}

    M_n_1::MS
    M_n_2::MS
    M_2n::MS
    M_mn_2n_1::MS
    M_mn_2n_2::MS
    U_n::UpperTriangular{T, MS}
    U_mn::UpperTriangular{T, MS}

    NextData::Union{TriDiagBlockDataNested, Nothing}

    next_idx::Vector{Int}
    next_x::MU

end

function initialize(N, m, n, P, A_list, B_list, level)

    I_separator = 1:(m+1):N

    LHS_A_list = zeros(P, n, n);
    LHS_B_list = zeros(P-1, n, n);

    MA_list = zeros(P-1, m*n, m*n);

    RHS = zeros(P * n);

    MA_chol = UpperTriangular(zeros(m*n, m*n));
    LHS_chol = UpperTriangular(zeros(P*n, P*n));

    factor_list = zeros(P-1, m*n, 2*n);

    M_n_1 = similar(A_list, n, n);
    M_n_2 = similar(A_list, n, n);
    U_mn = UpperTriangular(zeros(m*n, m*n));

    M_2n = similar(A_list, 2*n, 2*n);
    M_mn_2n_1 = zeros(m*n, 2*n);
    M_mn_2n_2 = zeros(m*n, 2*n);

    U_n = UpperTriangular(zeros(n, n));

    v_n = zeros(2*n);

    next_idx = Int[]

    for j = I_separator

        append!(next_idx, (j-1)*n+1:j*n)
        
    end

    next_x = zeros(P*n);

    data = TriDiagBlockDataNested(
        N, 
        m, 
        n, 
        P, 
        I_separator, 
        A_list, 
        B_list,
        LHS_A_list,
        LHS_B_list,
        MA_list,
        factor_list,
        RHS,
        MA_chol,
        LHS_chol,
        M_n_1,
        M_n_2,
        M_2n,
        M_mn_2n_1,
        M_mn_2n_2,
        U_n,
        U_mn,
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

        MA_list = zeros(P-1, m*n, m*n);

        RHS = zeros(P * n);

        MA_chol = UpperTriangular(zeros(m*n, m*n));
        LHS_chol = UpperTriangular(zeros(P*n, P*n));

        factor_list = zeros(P-1, m*n, 2*n);

        M_n_1 = similar(A_list, n, n);
        M_n_2 = similar(A_list, n, n);
        U_mn = UpperTriangular(zeros(m*n, m*n));

        M_2n = similar(A_list, 2*n, 2*n);
        M_mn_2n_1 = zeros(m*n, 2*n);
        M_mn_2n_2 = zeros(m*n, 2*n);

        U_n = UpperTriangular(zeros(n, n));

        v_n = zeros(2*n);

        next_idx = Int[]

        for j = I_separator

            append!(next_idx, (j-1)*n+1:j*n)
            
        end

        next_x = zeros(P*n);

        next_data = TriDiagBlockDataNested(
            N, 
            m, 
            n, 
            P, 
            I_separator,
            A_list, 
            B_list,
            LHS_A_list,
            LHS_B_list,
            MA_list,
            factor_list,
            RHS,
            MA_chol,
            LHS_chol,
            M_n_1,
            M_n_2,
            M_2n,
            M_mn_2n_1,
            M_mn_2n_2,
            U_n,
            U_mn,
            nothing,
            next_idx,
            next_x,
            );

        prev_data.NextData = next_data;
        prev_data = next_data;
    end

    return data

end


function cholesky_factorize(A_list, B_list, M_chol, A, B, U, N, n)

    copyto!(A, view(A_list, 1, :, :))
    cholesky!(Hermitian(A))
    copyto!(U, UpperTriangular(A))
    copyto!(view(M_chol, 1:n, 1:n), U.data)

    # Iterate over remaining blocks
    for i = 2:N

        # Solve for L_{i, i-1}
        copyto!(A, view(B_list, i-1, :, :))
        ldiv!(B, U', A)
        copyto!(view(M_chol, (i-1)*n-n+1:i*n-n, (i-1)*n +1:i*n), B)

        copyto!(A, view(A_list, i, :, :))

        # Compute Schur complement
        mul!(A, B', B, -1.0, 1.0)
        
        # Compute Cholesky factor for current block
        cholesky!(Hermitian(A));
        copyto!(U, UpperTriangular(A));
        copyto!(view(M_chol, (i-1)*n+1:i*n, (i-1)*n +1:i*n), U.data)

    end

end

function factorize(
    data::TriDiagBlockDataNested
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
MA_chol_list = data.MA_chol_list

MA_chol = data.MA_chol

A = data.M_n_1
B = data.M_n_2
M_2n = data.M_2n
M_mn_2n_1 = data.M_mn_2n_1
M_mn_2n_2 = data.M_mn_2n_2

U = data.U_n

@views for i = 1:P-1 #TODO get rid of views

    # Compute inverse of block tridiagonal matrices to compute inverse of MA (top left of Schur complement)
    cholesky_factorize(
        A_list[I_separator[i]+1:I_separator[i]+m, :, :], 
        B_list[I_separator[i]+1:I_separator[i]+m-1, :, :], 
        MA_chol,
        A,
        B,
        U,
        m, 
        n)

    MA_chol_list[i, :, :] .= MA_chol
    
    M_mn_2n_1[1:n, 1:n] .= B_list[I_separator[i], :, :]'
    M_mn_2n_1[m*n-n+1:m*n, n+1:2*n] .= B_list[I_separator[i+1]-1, :, :]

    M_mn_2n_2 .= M_mn_2n_1

    ldiv!(MA_chol', M_mn_2n_2)
    ldiv!(MA_chol, M_mn_2n_2)

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

    LHS_chol = data.LHS_chol

    cholesky_factorize(LHS_A_list, LHS_B_list, LHS_chol, A, B, U, P, n)

else

    data.NextData.A_list = LHS_A_list
    data.NextData.B_list = LHS_B_list
    factorize(data.NextData)

end

end

function solve(
    data::TriDiagBlockDataNested,
    d,
    x
)

P = data.P
n = data.n

I_separator = data.I_separator

B_list = data.B_list

MA_chol_list = data.MA_chol_list

factor_list = data.factor_list

B = data.M_n_2
U = data.U_mn
M_mn_2n_1 = data.M_mn_2n_1

RHS = data.RHS

# Assign RHS from d
for j = 1:P

    copyto!(view(RHS, (j-1)*n+1:j*n),  view(d, I_separator[j]*n-n+1:I_separator[j]*n)) # d_list[I_separator[j], :]

end

# Compute RHS from Schur complement
for i = 1:P-1

    copyto!(M_mn_2n_1, view(factor_list, i, :, :))
    mul!(view(RHS, (i-1)*n+1:(i+1)*n), M_mn_2n_1', view(d, I_separator[i]*n+1:I_separator[i+1]*n-n), -1.0, 1.0)

end

# RHS = invLHS * RHS; #TODO lmul! which is faster?

if isnothing(data.NextData)

    LHS_chol = data.LHS_chol
    ldiv!(LHS_chol', RHS)
    ldiv!(LHS_chol, RHS)

    # Assign RHS to x solution for separators
    for i = 1:P

        copyto!(view(x, I_separator[i]*n-n+1:I_separator[i]*n), view(RHS, (i-1)*n+1:i*n))
    
    end

else

    copyto!(data.next_x, view(x, data.next_idx))
    solve(data.NextData, RHS, data.next_x)
    copyto!(view(x, data.next_idx), data.next_x)

end

# Update d after Schur solve
for j = 1:P-1 #TODO remove B

    copyto!(B, view(B_list, I_separator[j], :, :)')
    mul!(view(d, I_separator[j]*n+1:I_separator[j]*n+n,), B, view(x, I_separator[j]*n-n+1:I_separator[j]*n), -1.0, 1.0)

    copyto!(B, view(B_list, I_separator[j+1]-1, :, :))
    mul!(view(d, I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n), B, view(x, I_separator[j+1]*n-n+1:I_separator[j+1]*n), -1.0, 1.0)

end

# solve for non-separators
for i = 1:P-1

    copyto!(U, view(MA_chol_list, i, :, :))
    ldiv!(U', view(d, I_separator[i]*n+1:I_separator[i+1]*n-n))
    ldiv!(U, view(d, I_separator[i]*n+1:I_separator[i+1]*n-n))
    copyto!(view(x, I_separator[i]*n+1:I_separator[i+1]*n-n), view(d, I_separator[i]*n+1:I_separator[i+1]*n-n))

end

end

end