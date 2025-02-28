using CUDA
using LinearAlgebra

import Pkg
include("src/TriDiagBlockCUDA.jl")
import .TriDiagBlockCUDA: TriDiagBlockDataCUDA, factorize, solve

n = 100 # size of each block
m = 2 # number of blocks between separators
N = 55 # number of diagonal blocks
P = Int((N + m) / (m+1))

#######################################
A_list = CuArray(zeros(N, n, n))
for i = 1:N
    temp = CuArray(randn(Float64, n, n))
    A_list[i, :, :] .= temp * temp' + n * I
end

B_list = CuArray(zeros(N-1, n, n))
for i = 1:N-1
    B_list[i, :, :] .= CuArray(randn(Float64, n, n))
end

x_true = CuArray(rand(N, n))
d_list = CuArray(zeros(N, n))

d_list[1, :] .= A_list[1, :, :] * x_true[1, :] + B_list[1, :, :] * x_true[2, :]

@views for i = 2:N-1
    d_list[i, :] .= B_list[i-1, :, :]' * x_true[i-1, :] + A_list[i, :, :] * x_true[i, :] + B_list[i, :, :] * x_true[i+1, :]
end

d_list[N, :] .= B_list[N-1, :, :]' * x_true[N-1, :] + A_list[N, :, :] * x_true[N, :]

d = CuArray(zeros(N * n))

@views for i = 1:N
    d[(i-1)*n+1:i*n] .= d_list[i, :]
end

x_true = reshape(x_true', N*n)

#################################################
level = 3
I_separator = collect(1:(m+1):N)

LHS_A_list = CuArray(zeros(P, n, n))
LHS_B_list = CuArray(zeros(P-1, n, n))

MA_list = CuArray(zeros(P-1, m*n, m*n))
RHS = CuArray(zeros(P * n))

MA_chol = CuUpperTriangular(CuArray(zeros(m*n, m*n)))
LHS_chol = CuUpperTriangular(CuArray(zeros(P*n, P*n)))

factor_list = CuArray(zeros(P-1, m*n, 2*n))

M_n_1 = similar(A_list, n, n)
M_n_2 = similar(A_list, n, n)
U_mn = CuUpperTriangular(CuArray(zeros(m*n, m*n)))

M_2n = similar(A_list, 2*n, 2*n)
M_mn_2n_1 = CuArray(zeros(m*n, 2*n))
M_mn_2n_2 = CuArray(zeros(m*n, 2*n))

U_n = CuUpperTriangular(CuArray(zeros(n, n)))

next_idx = CuArray(Int[])
for j in I_separator
    append!(next_idx, (j-1)*n+1:j*n)
end

next_x = CuArray(zeros(P*n))

data = TriDiagBlockDataNestedCUDA(
    N, m, n, P, I_separator, A_list, B_list, 
    LHS_A_list, LHS_B_list, MA_list, factor_list, RHS,
    MA_chol, LHS_chol, M_n_1, M_n_2, M_2n, M_mn_2n_1, 
    M_mn_2n_2, U_n, U_mn, nothing, next_idx, next_x
)

prev_data = data

for i = 2:level
    N = P
    P = Int((N + m) / (m+1))
    I_separator = collect(1:(m+1):N)

    LHS_A_list = CuArray(zeros(P, n, n))
    LHS_B_list = CuArray(zeros(P-1, n, n))
    MA_list = CuArray(zeros(P-1, m*n, m*n))
    RHS = CuArray(zeros(P * n))

    MA_chol = CuUpperTriangular(CuArray(zeros(m*n, m*n)))
    LHS_chol = CuUpperTriangular(CuArray(zeros(P*n, P*n)))

    factor_list = CuArray(zeros(P-1, m*n, 2*n))

    M_n_1 = similar(A_list, n, n)
    M_n_2 = similar(A_list, n, n)
    U_mn = CuUpperTriangular(CuArray(zeros(m*n, m*n)))

    M_2n = similar(A_list, 2*n, 2*n)
    M_mn_2n_1 = CuArray(zeros(m*n, 2*n))
    M_mn_2n_2 = CuArray(zeros(m*n, 2*n))

    U_n = CuUpperTriangular(CuArray(zeros(n, n)))

    next_idx = CuArray(Int[])
    for j in I_separator
        append!(next_idx, (j-1)*n+1:j*n)
    end

    next_x = CuArray(zeros(P*n))

    next_data = TriDiagBlockDataNestedCUDA(
        N, m, n, P, I_separator, A_list, B_list, 
        LHS_A_list, LHS_B_list, MA_list, factor_list, RHS,
        MA_chol, LHS_chol, M_n_1, M_n_2, M_2n, M_mn_2n_1, 
        M_mn_2n_2, U_n, U_mn, nothing, next_idx, next_x
    )

    prev_data.NextData = next_data
    prev_data = next_data
end

@time factorize(data)

x = CuArray(zeros(data.N * n))

@time solve(data, d, x)

norm(x - x_true)
