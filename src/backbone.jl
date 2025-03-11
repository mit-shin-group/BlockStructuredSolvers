function cholesky_factorize!(A_list, B_list, N)

    mypotrf!('U', A_list[1]) #TODO Allocation is here

    for i = 2:N

        mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[i-1], B_list[i-1])
        mygemm!('T', 'N', -1.0, B_list[i-1], B_list[i-1], 1.0, A_list[i])
        mypotrf!('U', A_list[i]) #TODO Allocation is here

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d_list::M, N, n) where {T, M<:Vector{AbstractVector{T}}}

    mytrsv!('U', 'T', 'N', M_chol_A_list[1], d_list[1]);

    for i = 2:N #TODO pprof.jl

        mygemv!('T', -1.0, M_chol_B_list[i-1], d_list[i-1], 1.0, d_list[i])

        mytrsv!('U', 'T', 'N', M_chol_A_list[i], d_list[i])

    end

    mytrsv!('U', 'N', 'N', M_chol_A_list[N], d_list[N])

    for i = N-1:-1:1

        mygemv!('N', -1.0, M_chol_B_list[i], d_list[i+1], 1.0, d_list[i])

        mytrsv!('U', 'N', 'N', M_chol_A_list[i], d_list[i])

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d_list::M, N, n) where {T, M<:Vector{AbstractMatrix{T}}}

    mytrsm!('L', 'U', 'T', 'N', 1.0, M_chol_A_list[1], d_list[1]);

    for i = 2:N

        mygemm!('T', 'N', -1.0, M_chol_B_list[i-1], d_list[i-1], 1.0, d_list[i])
        mytrsm!('L', 'U', 'T', 'N', 1.0, M_chol_A_list[i], d_list[i])

    end

    mytrsm!('L', 'U', 'N', 'N', 1.0, M_chol_A_list[N], d_list[N])

    for i = N-1:-1:1

        mygemm!('N', 'N', -1.0, M_chol_B_list[i], d_list[i+1], 1.0, d_list[i])
        mytrsm!('L', 'U', 'N', 'N', 1.0, M_chol_A_list[i], d_list[i])

    end

end