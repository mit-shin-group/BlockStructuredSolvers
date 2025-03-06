struct BlockTriDiagData_sequential_cholesky_factor{
    T, 
    MT <: AbstractArray{T, 3}}

    N::Int
    n::Int

    A_list::MT
    B_list::MT

end

function initialize_sequential_cholesky_factor(N, n, A_list, B_list)

    data = BlockTriDiagData_sequential_cholesky_factor(
        N,
        n,
        A_list,
        B_list
    )

    return data

end


function cholesky_factorize!(A_list, B_list, N)

    potrf!('U', view(A_list, :, :, 1))

    for i = 2:N

        trsm!('L', 'U', 'T', 'N', 1.0, view(A_list, :, :, i-1), view(B_list, :, :, i-1))
        gemm!('T', 'N', -1.0, view(B_list, :, :, i-1), view(B_list, :, :, i-1), 1.0, view(A_list, :, :, i))
        potrf!('U', view(A_list, :, :, i))

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 1}}

    trsv!('U', 'T', 'N', view(M_chol_A_list, :, :, 1), view(d, 1:n));

    for i = 2:N #TODO pprof.jl

        gemv!('T', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n), 1.0, view(d, (i-1)*n+1:i*n))

        trsv!('U', 'T', 'N',  view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n))

    end

    trsv!('U', 'N', 'N', view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n));

    for i = N-1:-1:1

        gemv!('N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n), 1.0, view(d, (i-1)*n+1:i*n))

        trsv!('U', 'N', 'N', view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n))

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 2}}

    trsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, 1), view(d, 1:n, :));

    for i = 2:N

        gemm!('T', 'N', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        trsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

    trsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n, :));

    for i = N-1:-1:1

        gemm!('N', 'N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        trsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

end

function factorize_sequential_cholesky_factor!(
    data::BlockTriDiagData_sequential_cholesky_factor
)

N = data.N

A_list = data.A_list
B_list = data.B_list

cholesky_factorize!(A_list, B_list, N)

end

function solve_sequential_cholesky_factor!(data::BlockTriDiagData_sequential_cholesky_factor, d, x)

    N = data.N
    n = data.n
    A_list = data.A_list
    B_list = data.B_list

    cholesky_solve!(A_list, B_list, d, N, n)

    x .= d

    return nothing
end