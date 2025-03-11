function cholesky_factorize!(A_list, B_list, N)

    mypotrf!('U', view(A_list, :, :, 1))

    for i = 2:N

        mytrsm!('L', 'U', 'T', 'N', 1.0, view(A_list, :, :, i-1), view(B_list, :, :, i-1))
        mygemm!('T', 'N', -1.0, view(B_list, :, :, i-1), view(B_list, :, :, i-1), 1.0, view(A_list, :, :, i))
        mypotrf!('U', view(A_list, :, :, i))

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 1}}

    mytrsv!('U', 'T', 'N', view(M_chol_A_list, :, :, 1), view(d, 1:n));

    for i = 2:N #TODO pprof.jl

        mygemv!('T', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n), 1.0, view(d, (i-1)*n+1:i*n))

        mytrsv!('U', 'T', 'N',  view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n))

    end

    mytrsv!('U', 'N', 'N', view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n));

    for i = N-1:-1:1

        mygemv!('N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n), 1.0, view(d, (i-1)*n+1:i*n))

        mytrsv!('U', 'N', 'N', view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n))

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d::M, N, n) where {T, M<:AbstractArray{T, 2}}

    mytrsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, 1), view(d, 1:n, :));

    for i = 2:N

        mygemm!('T', 'N', -1.0, view(M_chol_B_list, :, :, i-1), view(d, (i-2)*n+1:(i-1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        mytrsm!('L', 'U', 'T', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

    mytrsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, N), view(d, (N-1)*n+1:N*n, :));

    for i = N-1:-1:1

        mygemm!('N', 'N', -1.0, view(M_chol_B_list, :, :, i), view(d, i*n+1:(i+1)*n, :), 1.0, view(d, (i-1)*n+1:i*n, :))
        mytrsm!('L', 'U', 'N', 'N', 1.0, view(M_chol_A_list, :, :, i), view(d, (i-1)*n+1:i*n, :))

    end

end