struct BlockTriDiagData_sequential_cholesky_factor{ #TODO create initialize function
    T, 
    MT <: AbstractArray{T, 3},
    MS <: AbstractArray{T, 2},
    MU <: AbstractArray{T, 1}
    }

    N::Int
    n::Int

    A_list::MT
    B_list::MT

    M_chol_A_list::MT
    M_chol_B_list::MT

    M_n_1::MS
    M_n_2::MS

    v_n_1::MU
    v_n_2::MU

end

function initialize_sequential_cholesky_factor(N, n, A_list, B_list)

    M_chol_A_list = zeros(N, n, n);
    M_chol_B_list = zeros(N-1, n, n);

    v_n_1 = zeros(n);
    v_n_2 = zeros(n);

    M_n_1 = zeros(n, n);
    M_n_2 = zeros(n, n);

    data = BlockTriDiagData_sequential_cholesky_factor(
        N,
        n,
        A_list,
        B_list,
        M_chol_A_list,
        M_chol_B_list,
        M_n_1,
        M_n_2,
        v_n_1,
        v_n_2
    )

    return data

end


function cholesky_factorize!(A_list, B_list, M_chol_A_list, M_chol_B_list, A, B, N, n)

    copyto!(A, view(A_list, 1, :, :))
    # cholesky!(Hermitian(A))
    potrf!('U', A)
    copyto!(view(M_chol_A_list, 1, :, :), A)

    # Iterate over remaining blocks
    for i = 2:N

        # Solve for L_{i, i-1}
        copyto!(B, view(B_list, i-1, :, :))
        trtrs!('U', 'T', 'N', A, B)
        copyto!(view(M_chol_B_list, i-1, :, :), B)

        copyto!(A, view(A_list, i, :, :))

        # Compute Schur complement
        gemm!('T', 'N', -1.0, B, B, 1.0, A)
        
        # Compute Cholesky factor for current block
        # cholesky!(Hermitian(A));
        potrf!('U', A)
        copyto!(view(M_chol_A_list, i, :, :), A)

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::AbstractArray{T, 1}, A, u, v, N, n) where {T}
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n)

    trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

    trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n)

        gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n) .= v

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::AbstractArray{T, 2}, A, u, v, N, n) where {T}
    A .= view(M_chol_A_list, 1, :, :);
    v .= view(d, 1:n, :)

    trtrs!('U', 'T', 'N', A, v);
    view(d, 1:n, :) .= v;

    for i = 2:N

        A .= view(M_chol_B_list, i-1, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        gemm!('T', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        trtrs!('U', 'T', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

    trtrs!('U', 'N', 'N', A, v);
    view(d, (N-1)*n+1:N*n, :) .= v;

    for i = N-1:-1:1

        A .= view(M_chol_B_list, i, :, :);

        u .= v
        v .= view(d, (i-1)*n+1:i*n, :)

        gemm!('N', 'N', -1.0, A, u, 1.0, v)

        A .= view(M_chol_A_list, i, :, :);
        trtrs!('U', 'N', 'N', A, v)
        view(d, (i-1)*n+1:i*n, :) .= v

    end

end

function factorize_sequential_cholesky_factor!(
    data::BlockTriDiagData_sequential_cholesky_factor
)

N = data.N
n = data.n

A_list = data.A_list
B_list = data.B_list

M_chol_A_list = data.M_chol_A_list
M_chol_B_list = data.M_chol_B_list

A = data.M_n_1
B = data.M_n_2

cholesky_factorize!(A_list, B_list, M_chol_A_list, M_chol_B_list, A, B, N, n)

end

function solve_sequential_cholesky_factor!(data::BlockTriDiagData_sequential_cholesky_factor, d, x)

    N = data.N
    n = data.n

    M_chol_A_list = data.M_chol_A_list
    M_chol_B_list = data.M_chol_B_list

    A = data.M_n_1

    u = data.v_n_1
    v = data.v_n_2

    cholesky_solve!(M_chol_A_list, M_chol_B_list, d, A, u, v, N, n)

    x .= d

    return nothing
end