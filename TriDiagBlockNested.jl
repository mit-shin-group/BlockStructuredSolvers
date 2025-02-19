module TriDiagBlockNested

using LinearAlgebra

export TriDiagBlockDataNested, factorize, solve

mutable struct TriDiagBlockDataNested{
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
    U_B_list::MT

    LHS_A_list::MT
    LHS_B_list::MT
    LHS_U_B_list::MT

    invMA_list::MT

    invMA::MS
    invLHS::MS

    invMA_chol::UpperTriangular{T, MS}
    invLHS_chol::UpperTriangular{T, MS}

    A::MS
    B::MS
    C::MS
    D::MS
    E::MS

    L::LowerTriangular{T, MS}
    U::UpperTriangular{T, MS}

    D1::MU
    D2::MU
    D3::MU
    D4::MU

    NextData::Union{TriDiagBlockDataNested, Nothing}

end

function inverse_cholesky_factorize(A_list, B_list, U_B_list, invM_chol, invM, A, B, C, L, U, N, n)

    A .= view(A_list, 1, :, :)
    cholesky!(Hermitian(A))
    U .= UpperTriangular(A);
    L .= U';

    # invM_chol[1:n, 1:n] = U \ I
    LAPACK.trtri!('U', 'N', U.data)
    invM_chol[1:n, 1:n] .= U.data

    # Iterate over remaining blocks
    @views for i = 2:N

        # Solve for off diagonal of L inverse L_{i, i-1}
        # U_B_list[i-1, :, :] = L \ B_list[i-1, :, :]
        A .= B_list[i-1, :, :]
        ldiv!(B, L, A)
        U_B_list[i-1, :, :] .= B
        
        # Compute Schur complement
        # A .= A_list[i, :, :] - U_B_list[i-1, :, :]' * U_B_list[i-1, :, :]
        A .= A_list[i, :, :]
        mul!(A, B', B, -1.0, 1.0)
        
        # Compute Cholesky factor for current block
        cholesky!(Hermitian(A));
        U .= UpperTriangular(A);
        L .= U';

        LAPACK.trtri!('U', 'N', U.data)
        invM_chol[(i-1)*n+1:i*n, (i-1)*n +1:i*n] .= U.data

    end

    C .= A

    @views for level = 1:N-1

        for j = 1:N-level

            # temp = - U_B_list[j, :, :] * invM_chol[(j)*n+1:(j+1)*n, (j+level-1)*n+1:(j+level)*n]
            # temp = invM_chol[(j-1)*n+1:(j)*n, (j-1)*n+1:(j)*n] * temp
            # invM_chol[(j-1)*n+1:j*n, (j+level-1)*n +1:(j+level-1)*n+n] = temp

            B .=  U_B_list[j, :, :]
            A .= invM_chol[(j)*n+1:(j+1)*n, (j+level-1)*n+1:(j+level)*n]

            mul!(C, B, A)
            mul!(C, B, A, 0.0, -1.0)
            
            B .= invM_chol[(j-1)*n+1:(j)*n, (j-1)*n+1:(j)*n]

            mul!(A, B, C)

            invM_chol[(j-1)*n+1:j*n, (j+level-1)*n +1:(j+level-1)*n+n] .= A

        end

    end

    # invM .= Symmetric(invM_chol * invM_chol')
    mul!(invM, invM_chol, invM_chol');

end

function factorize(
    data::TriDiagBlockDataNested
)

N = data.N
P = data.P
n = data.n
m = data.m

I_separator = data.I_separator

A_list = data.A_list
B_list = data.B_list
U_B_list = data.U_B_list
LHS_A_list = data.LHS_A_list
LHS_B_list = data.LHS_B_list
LHS_U_B_list = data.LHS_U_B_list

invMA_list = data.invMA_list
invMA_chol = data.invMA_chol
invMA = data.invMA


A = data.A
B = data.B
C = data.C

L = data.L
U = data.U

@views for i = 1:P-1

    # Compute inverse of block tridiagonal matrices to compute inverse of MA (top left of Schur complement)
    inverse_cholesky_factorize(
        A_list[I_separator[i]+1:I_separator[i]+m, :, :], 
        B_list[I_separator[i]+1:I_separator[i]+m-1, :, :], 
        U_B_list, 
        invMA_chol,
        invMA,
        A,
        B,
        C,
        L,
        U,
        m, 
        n)
    
    invMA_list[i, :, :] .= invMA; #TODO how to store? invMA vs invMA_chol

    # LHS_A_list[i, :, :] -= B_list[I_separator[i], :, :] * invMA[1:n, 1:n] *  B_list[I_separator[i], :, :]'
    # LHS_A_list[i+1, :, :] -= B_list[I_separator[i+1]-1, :, :]' * invMA[m*n-n+1:m*n, m*n-n+1:m*n] * B_list[I_separator[i+1]-1, :, :]
    # LHS_B_list[i, :, :] -= B_list[I_separator[i], :, :] * invMA[1:n, m*n-n+1:m*n] *  B_list[I_separator[i+1]-1, :, :]
    LHS_A_list[i, :, :] .+= A_list[I_separator[i], :, :]

    B .= B_list[I_separator[i], :, :]
    A .= invMA[1:n, 1:n]
    mul!(C, B, A)
    A .= LHS_A_list[i, :, :]
    mul!(A, C, B', -1.0, 1.0)
    LHS_A_list[i, :, :] .= A

    B .= B_list[I_separator[i+1]-1, :, :]
    A .= invMA[m*n-n+1:m*n, m*n-n+1:m*n]
    mul!(C, A, B)
    A .= LHS_A_list[i+1, :, :]
    mul!(A, B', C, -1.0, 1.0)
    LHS_A_list[i+1, :, :] .= A

    A .= invMA[1:n, m*n-n+1:m*n]
    mul!(C, A, B)
    A .= LHS_B_list[i, :, :]
    B .= B_list[I_separator[i], :, :]
    mul!(A, B, C, -1.0, 1.0)
    LHS_B_list[i, :, :] .= A

end

A .= view(LHS_A_list, P, :, :)
A .+= view(A_list, I_separator[P], :, :)
LHS_A_list[P, :, :] .= A

if isnothing(data.NextData)

    invLHS_chol = data.invLHS_chol
    invLHS = data.invLHS
    inverse_cholesky_factorize(LHS_A_list, LHS_B_list, LHS_U_B_list, invLHS_chol, invLHS, A, B, C, L, U, P, n)

else

    data.NextData.A_list = LHS_A_list
    data.NextData.B_list = LHS_B_list
    factorize(data.NextData)

end

end

function solve(
    data::TriDiagBlockDataNested,
    d,
    RHS,
    x
)

# N = data.N
P = data.P
n = data.n
m = data.m

I_separator = data.I_separator

B_list = data.B_list

invMA_list = data.invMA_list

invLHS_chol = data.invLHS_chol

# A = data.A
B = data.B
# C = data.C
D = data.D
E = data.E
D1 = data.D1
D2 = data.D2
D3 = data.D3
D4 = data.D4

# Assign RHS from d
@views for j = 1:P

    RHS[(j-1)*n+1:j*n] .= d[I_separator[j]*n-n+1:I_separator[j]*n] # d_list[I_separator[j], :]

end

# Compute RHS from Schur complement
@views for i = 1:P-1

    # RHS[(i-1)*n+1:(i-1)*n+n] -= B_list[I_separator[i], :, :] * invMA_list[i, :, :][1:n, :] * d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    D .= invMA_list[i, :, :][1:n, :]
    B .= B_list[I_separator[i], :, :]

    D1 .= d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    D3 .= RHS[(i-1)*n+1:(i-1)*n+n]

    mul!(D2, D, D1)
    mul!(D3, B, D2, -1.0, 1.0)

    RHS[(i-1)*n+1:(i-1)*n+n] .= D3

    # RHS[(i-1)*n+n+1:(i-1)*n+n+n] -= B_list[I_separator[i+1]-1, :, :]' * invMA_list[i, :, :][m*n-n+1:m*n, :] * d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    D .= invMA_list[i, :, :][m*n-n+1:m*n, :]
    B .= B_list[I_separator[i+1]-1, :, :]

    D3 .= RHS[(i-1)*n+n+1:(i-1)*n+n+n]

    mul!(D2, D, D1)
    mul!(D3, B', D2, -1.0, 1.0)

    RHS[(i-1)*n+n+1:(i-1)*n+n+n] .= D3

end

# RHS = invLHS * RHS; #TODO lmul! which is faster?

if isnothing(data.NextData)

    invLHS_chol = data.invLHS_chol
    lmul!(invLHS_chol', RHS)
    lmul!(invLHS_chol, RHS)

    # Assign RHS to x solution for separators
    @views for i = 1:P

        x[I_separator[i]*n-n+1:I_separator[i]*n] .= RHS[(i-1)*n+1:i*n]

    end

    # println(x)

else

    seq = Int[]

    for i = I_separator

        append!(seq, (i-1)*n+1:i*n)
        
    end

    solve(data.NextData, RHS, zeros(data.NextData.P * n), view(x, seq))

end

# Update d after Schur solve
@views for j = 1:P-1
    copyto!(D3, d[I_separator[j]*n+1:I_separator[j]*n+n])
    copyto!(B, B_list[I_separator[j], :, :]')
    copyto!(D2, x[I_separator[j]*n-n+1:I_separator[j]*n])
    mul!(D3, B, D2, -1.0, 1.0)
    copyto!(d[I_separator[j]*n+1:I_separator[j]*n+n], D3)

    copyto!(D3, d[I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n])
    copyto!(B, B_list[I_separator[j+1]-1, :, :])
    copyto!(D2, x[I_separator[j+1]*n-n+1:I_separator[j+1]*n])
    mul!(D3, B, D2, -1.0, 1.0)
    copyto!(d[I_separator[j+1]*n-n-n+1:I_separator[j+1]*n-n], D3)

end

# solve for non-separators
@views for i = 1:P-1

    # x[I_separator[i]*n+1:I_separator[i+1]*n-n] = invMA_list[i, :, :] *  d[I_separator[i]*n+1:I_separator[i+1]*n-n]

    D1 .= d[I_separator[i]*n+1:I_separator[i+1]*n-n]
    D4 .= x[I_separator[i]*n+1:I_separator[i+1]*n-n]

    E .= invMA_list[i, :, :]

    mul!(D4, E, D1)

    x[I_separator[i]*n+1:I_separator[i+1]*n-n] .= D4

    
end


end

end