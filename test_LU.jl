using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, factorize!, solve

N = 97 # number of diagonal blocks
n = 16 # size of each block
P = 17 # number of separators
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

U_list = zeros(N, n, n);
L_list = zeros(N-1, n, n);

U_list[1, :, :] = A_list[1, :, :]

for i = 2:N
    
    L_list[i-1, :, :] = B_list[i-1, :, :]' * inv(U_list[i-1, :, :])
    U_list[i, :, :] = A_list[i, :, :] - L_list[i-1, :, :] * B_list[i-1, :, :]

end


M = zeros(N * n, N * n);

for i = 1:N
    M[(i-1)*n+1:i*n, (i-1)*n+1:i*n] = A_list[i, :, :]
end

for i = 1:N-1
    M[(i-1)*n+1:i*n, i*n+1:(i+1)*n] = B_list[i, :, :]
    M[i*n+1:(i+1)*n, (i-1)*n+1:i*n] = B_list[i, :, :]'
end


# Initialize storage for Cholesky factors
L_list = zeros(N, n, n)
L_B_list = zeros(N-1, n, n)

# Compute Cholesky factor for first block
L_list[1, :, :] = cholesky(A_list[1, :, :]).L

# Iterate over remaining blocks
for i = 2:N
    # Solve for L_{i, i-1}
    L_B_list[i-1, :, :] = B_list[i-1, :, :] * inv(L_list[i-1, :, :]')

    # Compute Schur complement
    Schur_complement = A_list[i, :, :] - L_B_list[i-1, :, :] * L_B_list[i-1, :, :]'

    # Compute Cholesky factor for current block
    L_list[i, :, :] = cholesky(Schur_complement).L
end