using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, factorize, solve

N = 97 # number of diagonal blocks
n = 8 # size of each block
P = 17 # number of separators
m  = trunc(Int, (N - P) / (P - 1))

#######################################
A_list = zeros(N, 8, 8)
for i = 1:N
    temp = rand(Float64, 8, 8)
    A_list[i, :, :] = temp * temp'

end

B_list = zeros(N-1, 8, 8)
for i = 1:N-1
    temp = rand(Float64, 8, 8)
    B_list[i, :, :] = temp * temp'
    # B_list[i, :, :] = Diagonal(ones(8)) * 0.01
end

xtrue_list = rand(N, n)
d_list = zeros(N, n)

d_list[1, :] = A_list[1, :, :] * xtrue_list[1, :] + B_list[1, :, :] * xtrue_list[2, :]

for i = 2:N-1

    d_list[i, :] = B_list[i-1, :, :]' * xtrue_list[i-1, :] + A_list[i, :, :] * xtrue_list[i, :] + B_list[i, :, :] * xtrue_list[i+1, :]

end

d_list[N, :] = B_list[N-1, :, :]' * xtrue_list[N-1, :] + A_list[N, :, :] * xtrue_list[N, :]


###########

batch_A_list = zeros(P-1, m, n, n)
batch_B_list = zeros(P-1, m-1, n, n)
temp_B_list = zeros(P-1, m-1, n, n)

MA = zeros((P-1) * m * n, (P-1) * m * n)
MB = zeros((P-1) * m * n, P * n)
MD = zeros(P * n, P * n)

data = TriDiagBlockData(N, m, n, P, 1:(m+1):N, A_list, B_list, batch_A_list, batch_B_list, temp_B_list, MA, MB, MD)

factorize(data)

x_list = zeros(N, n)
v = zeros((P-1) * m * n)
u = zeros(P * n)
batch_d_list = zeros(P-1, m, n)
temp_d_list = zeros(P-1, m, n)

solve(data, d_list, u, v, batch_d_list, temp_d_list, x_list)
