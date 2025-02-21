using LinearAlgebra

import Pkg
include("TriDiagBlockv3.jl")
import .TriDiagBlockv3: TriDiagBlockDatav3, factorize, solve

n = 10 # size of each block
P = 50 # number of separators
m = 3 # number of blocks between separators
N = P + (P - 1) * m # number of diagonal blocks

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

I_separator = 1:(m+1):N

LHS_A_list = zeros(P, n, n);
LHS_B_list = zeros(P-1, n, n);

MA_list = zeros(P-1, m*n, m*n);

LHS = zeros(P*n, P*n);
MA_chol = UpperTriangular(zeros(m*n, m*n));
LHS_chol = UpperTriangular(zeros(P*n, P*n));

factor_list = zeros(P-1, m*n, 2*n);

M_n_1 = similar(A_list, n, n);
M_n_2 = similar(A_list, n, n);
U_mn = UpperTriangular(zeros(m*n, m*n));

M_2n = similar(A_list, 2*n, 2*n);
M_mn_2n_1 = zeros(m*n, 2*n);
M_mn_2n_2 = zeros(m*n, 2*n);

U_n = UpperTriangular(zeros(n, n));

RHS = zeros(P * n);

data = TriDiagBlockDatav3(
    N, 
    m, 
    n, 
    P, 
    I_separator, 
    A_list, 
    B_list,
    LHS_A_list,
    LHS_B_list,
    MA_list,
    LHS,
    RHS,
    MA_chol,
    LHS_chol,
    factor_list,
    M_n_1,
    M_n_2,
    M_2n,
    M_mn_2n_1,
    M_mn_2n_2,
    U_n,
    U_mn,
    );

@time factorize(data);

x = zeros(N * n);

@time solve(data, d, x)

norm(x - x_true)