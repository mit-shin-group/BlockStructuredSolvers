using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, factorize, solve

N = 17 # number of diagonal blocks
n = 2 # size of each block
P = 5 # number of separators
m  = trunc(Int, (N - P) / (P - 1))

#######################################
A_list = zeros(N, n, n);
for i = 1:N
    temp = randn(Float64, n, n)
    A_list[i, :, :] = temp * temp' + n * I
end

B_list = zeros(N-1, n, n);
for i = 1:N-1
    temp = randn(Float64, n, n)
    B_list[i, :, :] = temp
end

x_true = rand(N, n);
d_list = zeros(N, n);
d_list[1, :] = A_list[1, :, :] * x_true[1, :] + B_list[1, :, :] * x_true[2, :];

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * x_true[i-1, :] + A_list[i, :, :] * x_true[i, :] + B_list[i, :, :] * x_true[i+1, :];

end

d_list[N, :] = B_list[N-1, :, :]' * x_true[N-1, :] + A_list[N, :, :] * x_true[N, :];
#################################################################

I_separator = 1:(m+1):N

####################################################################
MA = zeros((P-1) * m * n, (P-1) * m * n);
MB = zeros((P-1) * m * n, P * n);
MD = zeros(P * n, P * n);

@views for j = 1:P-1 #convert to while
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1:(j-1)*m*n+n] = A_list[I_separator[j]+1, :, :]
    MA[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*m*n+1+n:(j-1)*m*n+n+n] = B_list[I_separator[j]+1, :, :]
    for i = 2:m-1
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-2)*n+1:(j-1)*m*n+(i-1)*n] = B_list[I_separator[j]+i-1, :, :]'
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i)*n+1:(j-1)*m*n+(i+1)*n] = B_list[I_separator[j]+i, :, :]
        MA[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n, (j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = A_list[I_separator[j]+i, :, :]
    end
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-2)*n+1:(j-1)*m*n+(m-1)*n] = B_list[I_separator[j]+m-1, :, :]'
    MA[(j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n, (j-1)*m*n+(m-1)*n+1:(j-1)*m*n+m*n] = A_list[I_separator[j]+m, :, :]

    MB[(j-1)*m*n+1:(j-1)*m*n+n, (j-1)*n+1:(j-1)*n+n] = B_list[I_separator[j], :, :]'
    MB[j*m*n-n+1:j*m*n, (j-1)*n+n+1:(j-1)*n+n+n] = B_list[I_separator[j+1]-1, :, :]
end

@views for j = 1:P

    MD[(j-1)*n+1:j*n, (j-1)*n+1:j*n] = A_list[I_separator[j], :, :]

end

##################
v = zeros((P-1) * m * n)

@views for j = 1:P-1

    for i = 1:m
        v[(j-1)*m*n+(i-1)*n+1:(j-1)*m*n+i*n] = d_list[I_separator[j]+i, :]
    end

end
####################################
u = zeros(P * n)

@views for j = 1:P

    u[(j-1)*n+1:j*n] = d_list[I_separator[j], :]

end


######
x_list = zeros(N, n);
x_separator_sol = inv(MD - MB' * inv(MA) * MB) * (u - MB' * inv(MA) * v);

x_list[I_separator, :] = reshape(x_separator_sol, n, P)';



############
using LinearAlgebra, SparseArrays

# Function to compute Bunch-Kaufman factorization for block tridiagonal matrix

function block_bunchkaufman(A::Matrix{Float64}, block_size::Int)
    n = size(A, 1) ÷ block_size  # Number of blocks
    L = zeros(size(A))
    D = zeros(size(A))
    P = I(size(A, 1))  # Identity permutation matrix

    for i in 1:n
        idx = ((i-1)*block_size+1):(i*block_size)  # Block indices

        # Apply Bunch-Kaufman factorization on block B_i
        F = bunchkaufman(Hermitian(A[idx, idx]))
        
        # Store L and D blocks
        L[idx, idx] = F.U
        D[idx, idx] = Diagonal(F.D)
        P[idx, idx] = Matrix(F.P)

        # Eliminate below the diagonal
        if i < n
            next_idx = (i*block_size+1):((i+1)*block_size)
            L[next_idx, idx] = A[next_idx, idx] * inv(D[idx, idx] * L[idx, idx]')
            A[next_idx, next_idx] -= L[next_idx, idx] * D[idx, idx] * L[next_idx, idx]'
        end
    end

    return L, D, P
end


# Example block tridiagonal matrix
block_size = 2
A = [
    4.0 1.0  1.0 0.0  0.0 0.0;
    1.0 3.0  0.5 1.0  0.0 0.0;
    1.0 0.5  3.0 2.0  1.0 0.0;
    0.0 1.0  2.0 5.0  1.5 1.0;
    0.0 0.0  1.0 1.5  4.0 2.0;
    0.0 0.0  0.0 1.0  2.0 3.0
]
A = deepcopy(MA[1:m*n, 1:m*n])
B = deepcopy(A)

# Compute block Bunch-Kaufman factorization
L, D, P = block_bunchkaufman(A, block_size)

# Verify decomposition: P * A * P' ≈ L * D * L'
println("P * A * P' ≈ L * D * L' ? ", isapprox(P * B * P', L * D * L'))