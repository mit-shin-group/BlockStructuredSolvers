using LinearAlgebra

import Pkg
include("TriDiagBlockNestedv3.jl")
import .TriDiagBlockNested: TriDiagBlockDataNested, initialize, factorize, solve

n = 10 # size of each block
# P = 17 # number of separators
m = 2 # number of blocks between separators
N = 55 # number of diagonal blocks
P = Int((N + m) / (m+1))

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

data = initialize(N, m, n, P, A_list, B_list, 1);

@time factorize(data);
x = zeros(data.N * n);

@time solve(data, d, x);

#################################################

M_chol_A_list = data.LHS_chol_A_list;
M_chol_B_list = data.LHS_chol_B_list;

NN = size(M_chol_A_list, 1)  # Number of block rows

# Initialize full matrix with zeros
M = zeros(NN * n, NN * n);

# Fill diagonal blocks A
for i in 1:NN
    row_idx = (i-1) * n + 1 : i * n
    M[row_idx, row_idx] = M_chol_A_list[i, :, :]  # Place A[i] on the diagonal
end

# Fill lower off-diagonal blocks B
for i in 1:NN-1
    col_idx = i * n + 1 : (i+1) * n
    row_idx = (i-1) * n + 1 : i * n
    M[row_idx, col_idx] = M_chol_B_list[i, :, :]  # Place B[i] in lower off-diagonal
end

H = repeat(data.RHS, 1, 2*n)
temp_result = M' \ H;
temp_result = M \ temp_result;

#################################################



A = data.M_n_1;
B = data.M_n_2;

d = H;
u = zeros(n, 2*n);
v = zeros(n, 2*n);

function cholesky_solve_matrix(M_chol_A_list, M_chol_B_list, d, A, u, v, N, n)
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n, :)

    LAPACK.trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n, :) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        BLAS.gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

    LAPACK.trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n, :) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        BLAS.gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

end


function cholesky_solve(M_chol_A_list, M_chol_B_list, d, A, u, v, N, n)
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n)

    LAPACK.trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        BLAS.gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

    LAPACK.trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        BLAS.gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        LAPACK.trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

end

@time cholesky_solve_matrix(M_chol_A_list, M_chol_B_list, d, A, u, v, 19, n)


d - temp_result


