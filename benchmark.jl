using LinearAlgebra, SparseArrays, BlockArrays

using LDLFactorizations
using HSL

import Pkg
include("TriDiagBlockNestedv2.jl")
import .TriDiagBlockNested: TriDiagBlockDataNested, initialize, factorize, solve

######

function construct_block_tridiagonal(A_list, B_list)
    N, n, _ = size(A_list)
    blocks = Matrix{Float64}[]

    # Construct the block matrix row-wise
    for i = 1:N
        row_blocks = Any[]
        for j = 1:N
            if i == j
                push!(row_blocks, A_list[i, :, :])  # Diagonal blocks
            elseif j == i + 1
                push!(row_blocks, B_list[i, :, :])  # Upper diagonal blocks
            elseif j == i - 1
                push!(row_blocks, B_list[j, :, :]' )  # Lower diagonal blocks (transpose)
            else
                push!(row_blocks, zeros(n, n))  # Zero blocks elsewhere
            end
        end
        push!(blocks, hcat(row_blocks...))
    end

    return vcat(blocks...)
end

n = 50 # size of each block
m = 2 # number of blocks between separators
N = 55 # number of diagonal blocks
P = Int((N + m) / (m+1)) # number of separators
level = 3; # number of nested level

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

@views for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * x_true[i-1, :] + A_list[i, :, :] * x_true[i, :] + B_list[i, :, :] * x_true[i+1, :];

end

d_list[N, :] = B_list[N-1, :, :]' * x_true[N-1, :] + A_list[N, :, :] * x_true[N, :];

d = zeros(N * n);

@views for i = 1:N
    
    d[(i-1)*n+1:i*n] = d_list[i, :]

end

x_true = reshape(x_true', N*n);

BigMatrix = construct_block_tridiagonal(A_list, B_list);


#################################################

BigMatrix_57 = Ma57(BigMatrix);

@time ma57_factorize!(BigMatrix_57);
@time x = ma57_solve(BigMatrix_57, d);

norm(x - x_true)

#################################################

@time LDLT = ldl(BigMatrix);  # LDLᵀ factorization of A

@time x = LDLT \ d;  # solves Ax = b

norm(x - x_true)

#*********************
data = initialize(N, m, n, P, A_list, B_list, level);

@time factorize(data);

x = zeros(data.N * n);

@time solve(data, d, x)

norm(x - x_true)