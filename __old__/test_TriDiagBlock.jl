import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, ThomasSolve

using LinearAlgebra

N = 100
n = 3

A_i = [[5.2478, -5.2896, 0] [-5.2896, 7.5183, -1.6938] [0, -1.6938, 1.3119]]
# Q_i = rand(n, n)
# Q_i = Q_i' * Q_i
A_list = repeat(A_i, 1, 1, N)
A_list = permutedims(A_list, (3, 1, 2))
temp_list = zeros(N, n, n)
# B_i = Diagonal(ones(n))
# B_i = zeros(n, n)
B_i = rand(n, n)
B_i = B_i' * B_i
B_list = repeat(B_i, 1, 1, N-1)
B_list = permutedims(B_list, (3, 1, 2))

xtrue_list = rand(N, n)
x_list = zeros(N, n)
d_list = zeros(N, n)

d_list[1, :] = A_list[1, :, :] * xtrue_list[1, :] + B_list[1, :, :] * xtrue_list[2, :]

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * xtrue_list[i-1, :] + A_list[i, :, :] * xtrue_list[i, :] + B_list[i, :, :] * xtrue_list[i+1, :]

end

d_list[N, :] = B_list[N-1, :, :]' * xtrue_list[N-1, :] + A_list[N, :, :] * xtrue_list[N, :]

data = TriDiagBlockData(N, n, A_list, B_list, B_list, d_list, temp_list, x_list)

ThomasSolve(data)

display(data.x_list - xtrue_list)