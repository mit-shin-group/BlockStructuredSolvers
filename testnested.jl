using LinearAlgebra

import Pkg
include("TriDiagBlockNested.jl")
import .TriDiagBlockNested: TriDiagBlockDataNested, factorize, solve

n = 10 # size of each block
P = 7 # number of separators
m = 2 # number of blocks between separators
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

#################################################
level = 2;

I_separator = 1:(m+1):N;

U_B_list = zeros(m-1, n, n);
LHS_A_list = zeros(P, n, n);
LHS_B_list = zeros(P-1, n, n);
LHS_U_B_list = zeros(P-1, n, n);

invMA_list = zeros(P-1, m*n, m*n);

invMA = zeros(m*n, m*n);
invLHS = zeros(P*n, P*n);
invMA_chol = UpperTriangular(zeros(m*n, m*n));
invLHS_chol = UpperTriangular(zeros(P*n, P*n));

A = similar(A_list, n, n);
B = similar(A_list, n, n);
C = similar(A_list, n, n);
D = similar(A_list, n, m*n);
E = similar(A_list, m*n, m*n);

D1 = similar(A_list, m*n);
D2 = similar(A_list, n);
D3 = similar(A_list, n);
D4 = similar(A_list, m*n);

L = LowerTriangular(zeros(n, n));
U = UpperTriangular(zeros(n, n));

data = TriDiagBlockDataNested(
    N, 
    m, 
    n, 
    P, 
    I_separator, 
    A_list, 
    B_list, 
    U_B_list,
    LHS_A_list,
    LHS_B_list,
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
    E,
    L,
    U,
    D1,
    D2,
    D3,
    D4,
    nothing,
    nothing
    );

prev_data = data;

for i = 2:level

    N = P;
    m = 2;
    P = 3;

    I_separator = 1:(m+1):N;

    A_list = zeros(P, n, n);
    B_list = zeros(P-1, n, n);

    U_B_list = zeros(m-1, n, n);
    LHS_A_list = zeros(P, n, n);
    LHS_B_list = zeros(P-1, n, n);
    LHS_U_B_list = zeros(P-1, n, n);

    invMA_list = zeros(P-1, m*n, m*n);

    invMA = zeros(m*n, m*n);
    invLHS = zeros(P*n, P*n);
    invMA_chol = UpperTriangular(zeros(m*n, m*n));
    invLHS_chol = UpperTriangular(zeros(P*n, P*n));

    A = similar(A_list, n, n);
    B = similar(A_list, n, n);
    C = similar(A_list, n, n);
    D = similar(A_list, n, m*n);
    E = similar(A_list, m*n, m*n);

    D1 = similar(A_list, m*n);
    D2 = similar(A_list, n);
    D3 = similar(A_list, n);
    D4 = similar(A_list, m*n);

    L = LowerTriangular(zeros(n, n));
    U = UpperTriangular(zeros(n, n));

    next_data = TriDiagBlockDataNested(
        N, 
        m, 
        n, 
        P, 
        I_separator, 
        A_list, 
        B_list, 
        U_B_list,
        LHS_A_list,
        LHS_B_list,
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
        E,
        L,
        U,
        D1,
        D2,
        D3,
        D4,
        prev_data,
        nothing
        );

    prev_data.NextData = next_data;
    prev_data = next_data;
end

@time factorize(data);

n = 10 # size of each block
P = 7 # number of separators
m = 2 # number of blocks between separators
N = P + (P - 1) * m # number of diagonal blocks

RHS = zeros(P * n);
x = zeros(N * n);

@time solve(data, d, RHS, x)

norm(x - x_true)

##################
# seq = Int[]

# for i = data.I_separator

#     append!(seq, (i-1)*n+1:i*n)
    
# end

# solve(data.NextData, view(d, seq), zeros(data.NextData.P * n), view(x, seq))