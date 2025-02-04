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
    temp = rand(Float64, n, n)
    A_list[i, :, :] = temp * temp'
end

B_list = zeros(N-1, n, n);
zero_list = zeros(N-1, n, n);
for i = 1:N-1
    temp = rand(Float64, n, n)
    B_list[i, :, :] = temp * temp'
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

######
batch_A_list = zeros(P-1, m, n, n);
batch_B_list = zeros(P-1, m-1, n, n);

@views for j = 1:P-1

    batch_A_list[j, :, :, :] = A_list[I_separator[j]+1:I_separator[j+1]-1, :, :]
    batch_B_list[j, :, :, :] = B_list[I_separator[j]+1:I_separator[j+1]-1-1, :, :]

end

temp_A_list = deepcopy(batch_A_list);
temp_B_list = deepcopy(batch_B_list);

##################################################################################################################################
ipiv = Vector{LinearAlgebra.BlasInt}(undef, n);
A = similar(temp_A_list, n, n);
B = similar(temp_B_list, n, n);
C = similar(temp_A_list, n, n);
D = similar(temp_A_list, n);

data = TriDiagBlockData(N, m, n, P, 1:(m+1):N, A_list, B_list, batch_A_list, batch_B_list, temp_A_list, temp_B_list, MA, MB, MD, A, B, C, D, ipiv);

@time factorize!(data)

x_list = zeros(N, n);
v = zeros((P-1) * m * n);
u = zeros(P * n);
temp_d_list = zeros(P-1, m, n);
batch_d_list = zeros(P-1, m, n)

@views for j = 1:P-1

    batch_d_list[j, :, :] = d_list[I_separator[j]+1:I_separator[j+1]-1, :]

end

@time solve(data, d_list, u, v, batch_d_list, temp_d_list, x_list)
