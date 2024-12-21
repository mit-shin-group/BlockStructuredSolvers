using LinearAlgebra

import Pkg
include("TriDiagBlock.jl")
import .TriDiagBlock: TriDiagBlockData, factorize, solve

N = 97 # number of diagonal blocks
n = 20 # size of each block
P = 17 # number of separators
m  = trunc(Int, (N - P) / (P - 1))

#######################################
A_list = Vector{Matrix{Float64}}();
for i = 1:N
    temp = rand(Float64, n, n);
    temp = temp * temp';
    temp = (temp + temp') / 2
    push!(A_list, temp);

end

B_list = Vector{Matrix{Float64}}();
for i = 1:N-1
    temp = rand(Float64, n, n);
    temp = temp * temp';
    temp = (temp + temp') / 2
    push!(B_list, temp);
end

x_true = rand(N, n);
xtrue_list = Vector{Vector{Float64}}();
for i = 1:N
    push!(xtrue_list, x_true[i, :]);
end

d_list = Vector{Vector{Float64}}();

push!(d_list, A_list[1] * xtrue_list[1] + B_list[1] * xtrue_list[2]);

for i = 2:N-1

    push!(d_list,  B_list[i-1]' * xtrue_list[i-1] + A_list[i] * xtrue_list[i] + B_list[i] * xtrue_list[i+1]);

end

push!(d_list, B_list[N-1]' * xtrue_list[N-1] + A_list[N] * xtrue_list[N]);
#################################################################

I_separator = 1:(m+1):N

####################################################################
MA = zeros((P-1) * m * n, (P-1) * m * n);
MB = zeros((P-1) * m * n, P * n);
MD = zeros(P * n, P * n);

######
batch_A_list = Vector{Vector{Matrix{Float64}}}();
batch_B_list = Vector{Vector{Matrix{Float64}}}();

@views for j = 1:P-1

    push!(batch_A_list, A_list[I_separator[j]+1:I_separator[j+1]-1])
    push!(batch_B_list, B_list[I_separator[j]+1:I_separator[j+1]-1-1])

end

temp_A_list = deepcopy(batch_A_list);
temp_B_list = deepcopy(batch_B_list);



##################################################################################################################################
ipiv = zeros(Int, n);
data = TriDiagBlockData(N, m, n, P, 1:(m+1):N, A_list, B_list, batch_A_list, batch_B_list, temp_A_list, temp_B_list, MA, MB, MD, ipiv);

@time factorize(data)

x_list = zeros(N, n);
v = zeros((P-1) * m * n);
u = zeros(P * n);
temp_d_list = zeros(P-1, m, n);
batch_d_list = Vector{Vector{Vector{Float64}}}();

@views for j = 1:P-1

    push!(batch_d_list, d_list[I_separator[j]+1:I_separator[j+1]-1])

end

@time solve(data, d_list, u, v, batch_d_list, temp_d_list, x_list)
