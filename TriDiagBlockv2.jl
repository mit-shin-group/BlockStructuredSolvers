module TriDiagBlockv2

using LinearAlgebra

export TriDiagBlockDatav2, factorize, solve

struct TriDiagBlockDatav2{
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

    U_A_list::MT
    U_B_list::MT

    LHS_A_list::MT
    LHS_B_list::MT

    LHS_U_A_list::MT
    LHS_U_B_list::MT

    invMA_list::MT

    invMA::Symmetric{T, MS}
    invLHS::Symmetric{T, MS}

    invMA_chol::UpperTriangular{T, MS}
    invLHS_chol::UpperTriangular{T, MS}

    A::MS
    B::MS
    C::MS

    D1::MU
    D2::MU

end

function inverse_cholesky_factorize(A_list, B_list, U_A_list, U_B_list, invM_chol, invM, A, N, n)

    copyto!(A, @view A_list[1, :, :])
    U_A_list[1, :, :] = cholesky!(Hermitian(A)).U

    # Iterate over remaining blocks
    for i = 2:N
        # Solve for L_{i, i-1}
        U_B_list[i-1, :, :] = U_A_list[i-1, :, :]' \  B_list[i-1, :, :]

        # Compute Schur complement
        Schur_complement = A_list[i, :, :] - U_B_list[i-1, :, :]' * U_B_list[i-1, :, :]

        # Compute Cholesky factor for current block
        U_A_list[i, :, :] = cholesky(Hermitian(Schur_complement)).U

    end

    for i = 1:N

        invM_chol[(i-1)*n+1:i*n, (i-1)*n +1:i*n] = U_A_list[i, :, :] \ I

        for level = 1:N-1

            for j = 1:N-level

                temp = -U_B_list[j, :, :] * invM_chol[(j)*n+1:(j+1)*n, (j+level-1)*n+1:(j+level)*n]
                temp = invM_chol[(j-1)*n+1:(j)*n, (j-1)*n+1:(j)*n] * temp
                invM_chol[(j-1)*n+1:j*n, (j+level-1)*n +1:(j+level-1)*n+n] = temp

            end

        end

    end

    copyto!(invM, Symmetric(invM_chol * invM_chol'))

end

function factorize(
    data::TriDiagBlockDatav2
)

N = data.N
P = data.P
n = data.n
m = data.m

I_separator = data.I_separator

A_list = data.A_list
B_list = data.B_list
U_A_list = data.U_A_list
U_B_list = data.U_B_list
LHS_A_list = data.LHS_A_list
LHS_B_list = data.LHS_B_list
LHS_U_A_list = data.LHS_U_A_list
LHS_U_B_list = data.LHS_U_B_list

invMA_list = data.invMA_list

invLHS_chol = data.invLHS_chol

invMA_chol = data.invMA_chol
invMA = data.invMA
invLHS = data.invLHS

A = data.A
B = data.B
C = data.C

@views for i = 1:P-1

    inverse_cholesky_factorize(
        A_list[I_separator[i]+1:I_separator[i]+m, :, :], 
        B_list[I_separator[i]+1:I_separator[i]+m-1, :, :], 
        U_A_list, 
        U_B_list, 
        invMA_chol,
        invMA,
        A, 
        m, 
        n)

    invMA_list[i, :, :] = invMA;

    LHS_A_list[i, :, :] -= B_list[I_separator[i], :, :] * invMA[1:n, 1:n] *  B_list[I_separator[i], :, :]'
    LHS_A_list[i+1, :, :] -= B_list[I_separator[i+1]-1, :, :]' * invMA[m*n-n+1:m*n, m*n-n+1:m*n] * B_list[I_separator[i+1]-1, :, :]
    LHS_B_list[i, :, :] -= B_list[I_separator[i], :, :] * invMA[1:n, m*n-n+1:m*n] *  B_list[I_separator[i+1]-1, :, :]
    LHS_A_list[i, :, :] += A_list[I_separator[i], :, :]

end

LHS_A_list[P, :, :] += A_list[I_separator[P], :, :];

inverse_cholesky_factorize(LHS_A_list, LHS_B_list, LHS_U_A_list, LHS_U_B_list, invLHS_chol, invLHS, A, P, n)

end

function solve(
    data::TriDiagBlockDatav2,
    d,
    RHS,
    x
)

N = data.N
P = data.P
n = data.n
m = data.m

I_separator = data.I_separator

B_list = data.B_list

invMA_list = data.invMA_list

# invLHS = data.invLHS
invLHS_chol = data.invLHS_chol

A = data.A
B = data.B
C = data.C
D1 = data.D1
D2 = data.D2

@views for j = 1:P

    copyto!(RHS[(j-1)*n+1:j*n], d[I_separator[j]*n-n+1:I_separator[j]*n, :]) # d_list[I_separator[j], :]

end

@views for i = 1:P-1

    RHS[(i-1)*n+1:(i-1)*n+n] -= B_list[I_separator[i], :, :] * invMA_list[i, :, :][1:n, :] * d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    RHS[(i-1)*n+n+1:(i-1)*n+n+n] -= B_list[I_separator[i+1]-1, :, :]' * invMA_list[i, :, :][m*n-n+1:m*n, :] * d[I_separator[i]*n+1:I_separator[i+1]*n-n]

end

# RHS = invLHS * RHS; #TODO lmul! which is faster?
lmul!(invLHS_chol', RHS)
lmul!(invLHS_chol, RHS)

@views for i = 1:P

    x[I_separator[i]*n-n+1:I_separator[i]*n] = RHS[(i-1)*n+1:i*n]

end

@views for j = 1:P-1
    copyto!(D1, d[I_separator[j]*n+1:I_separator[j]*n+n])
    copyto!(B, B_list[I_separator[j], :, :]')
    copyto!(D2, x[I_separator[j]*n-n+1:I_separator[j]*n])
    mul!(D1, B, D2, -1.0, 1.0)
    copyto!(d[I_separator[j]*n+1:I_separator[j]*n+n], D1)

    copyto!(D1, d[I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n])
    copyto!(B, B_list[I_separator[j+1]-1, :, :])
    copyto!(D2, x[I_separator[j+1]*n-n+1:I_separator[j+1]*n])
    mul!(D1, B, D2, -1.0, 1.0)
    copyto!(d[I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n], D1)

end

@views for i = 1:P-1

    x[I_separator[i]*n+1:I_separator[i+1]*n-n] = invMA_list[i, :, :] *  d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    
end

end

end