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

U_A_list = zeros(P-1, m, n, n);
U_B_list = zeros(P-1, m-1, n, n);
LHS_A_list = zeros(P, n, n);
LHS_B_list = zeros(P-1, n, n);
LHS_U_A_list = zeros(P, n, n);
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

data = TriDiagBlockDatav3(
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
    E,
    L,
    U,
    D1,
    D2,
    D3,
    D4
    );

@time factorize(data);

MRHS = zeros(P*n, P*n);

B = randn(n, n);

for i = 1:P-1
    MRHS[(i-1)*n+1:i*n, (i-1)*n+1:i*n] = LHS_U_A_list[i, :, :]
    MRHS[(i-1)*n+1:i*n, (i-1)*n+n+1:i*n+n] = LHS_U_B_list[i, :, :]
end

MRHS[(P-1)*n+1:P*n, (P-1)*n+1:P*n] = LHS_U_A_list[P, :, :]

MLHS = zeros(P*n, P*n);
MLHS[1:n, n+1:2*n] = B;

sol = MRHS' \ MLHS;
sol = MRHS \ sol;

temp = zeros(P * n, n);

# copyto!(U, view(LHS_U_A_list, 1, :, :))
# ldiv!(A, U', B)
# copyto!(view(temp, 1:n, :), A)

# for i = 1:P-1

#     mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 0.0)

#     copyto!(U, view(LHS_U_A_list, i+1, :, :))
#     ldiv!(A, U', B)
#     copyto!(view(temp, i*n+1:(i+1)*n, :), A)

# end

# copyto!(B, view(temp, (P-1)*n+1:P*n, :))
# copyto!(U, view(LHS_U_A_list, P, :, :))
# ldiv!(A, U, B)
# copyto!(view(temp, (P-1)*n+1:P*n, :), A)

# for i = P-1:-1:1

#     copyto!(B, view(temp, (i-1)*n+1:i*n, :))
#     mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 1.0)

#     copyto!(U, view(LHS_U_A_list, i, :, :))
#     ldiv!(A, U, B)
#     copyto!(view(temp, (i-1)*n+1:i*n, :), A)

# end

# function cholesky_solve(temp, LHS_U_A_list, LHS_U_B_list, B, U, A, P, n)

#     copyto!(U, view(LHS_U_A_list, 1, :, :))
#     ldiv!(A, U', B)
#     copyto!(view(temp, 1:n, :), A)

#     for i = 1:P-1

#         mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 0.0)

#         copyto!(U, view(LHS_U_A_list, i+1, :, :))
#         ldiv!(A, U', B)
#         copyto!(view(temp, i*n+1:(i+1)*n, :), A)

#     end

#     copyto!(B, view(temp, (P-1)*n+1:P*n, :))
#     copyto!(U, view(LHS_U_A_list, P, :, :))
#     ldiv!(A, U, B)
#     copyto!(view(temp, (P-1)*n+1:P*n, :), A)

#     for i = P-1:-1:1

#         copyto!(B, view(temp, (i-1)*n+1:i*n, :))
#         mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 1.0)

#         copyto!(U, view(LHS_U_A_list, i, :, :))
#         ldiv!(A, U, B)
#         copyto!(view(temp, (i-1)*n+1:i*n, :), A)

#     end

# end

function cholesky_solve(temp, LHS_U_A_list, LHS_U_B_list, B, U, A, P, n)

    copyto!(U, view(LHS_U_A_list, 1, :, :))
    ldiv!(A, U', B)
    copyto!(view(temp, 1:n, :), A)

    for i = 1:P-1

        mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 0.0)

        copyto!(U, view(LHS_U_A_list, i+1, :, :))
        ldiv!(A, U', B)
        copyto!(view(temp, i*n+1:(i+1)*n, :), A)

    end

    copyto!(B, view(temp, (P-1)*n+1:P*n, :))
    copyto!(U, view(LHS_U_A_list, P, :, :))
    ldiv!(A, U, B)
    copyto!(view(temp, (P-1)*n+1:P*n, :), A)

    for i = P-1:-1:1

        copyto!(B, view(temp, (i-1)*n+1:i*n, :))
        mul!(B, view(LHS_U_B_list, i, :, :), A, -1.0, 1.0)

        copyto!(U, view(LHS_U_A_list, i, :, :))
        ldiv!(A, U, B)
        copyto!(view(temp, (i-1)*n+1:i*n, :), A)

    end

end

cholesky_solve(temp, LHS_U_A_list, LHS_U_B_list, B, U, A, P, n)