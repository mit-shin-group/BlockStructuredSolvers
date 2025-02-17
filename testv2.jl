using LinearAlgebra

import Pkg
include("TriDiagBlockv2.jl")
import .TriDiagBlockv2: TriDiagBlockDatav2, factorize, solve

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

U_A_list = zeros(m, n, n);
U_B_list = zeros(m-1, n, n);
LHS_A_list = zeros(P, n, n);
LHS_B_list = zeros(P-1, n, n);
LHS_U_A_list = zeros(P, n, n);
LHS_U_B_list = zeros(P-1, n, n);

invMA_list = zeros(P-1, m*n, m*n);

invMA = Symmetric(zeros(m*n, m*n));
invLHS = Symmetric(zeros(P*n, P*n));
invMA_chol = UpperTriangular(zeros(m * n, m * n));
invLHS_chol = UpperTriangular(zeros(P*n, P*n));

A = similar(A_list, n, n);
B = similar(A_list, n, n);
C = similar(A_list, n, n);
D = similar(A_list, n, m*n);

D1 = similar(A_list, m*n);
D2 = similar(A_list, n);
D3 = similar(A_list, n);

F1 = cholesky(Matrix{Float64}(I, n, n))
L1 = LowerTriangular(zeros(n, n));
U1 = UpperTriangular(zeros(n, n));

F2 = cholesky(Matrix{Float64}(I, n, n))
L2 = LowerTriangular(zeros(n, n));
U2 = UpperTriangular(zeros(n, n));

data = TriDiagBlockDatav2(
    N, 
    m, 
    n, 
    P, 
    I_separator, 
    A_list, 
    B_list, 
    U_A_list, 
    U_B_list,
    LHS_A_list,
    LHS_B_list,
    LHS_U_A_list,
    LHS_U_B_list,
    invMA_list,
    invMA,
    invLHS,
    invMA_chol,
    invLHS_chol,
    A,
    B,
    C,
    D,
    L1,
    U1,
    L2,
    U2,
    F1,
    F2,
    D1,
    D2,
    D3
    );

@time factorize(data);

RHS = zeros(P * n);
x = zeros(N * n);

@time solve(data, d, RHS, x)

norm(x - x_true)