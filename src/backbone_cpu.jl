function cholesky_factorize!(A_list, B_list, N)

    mypotrf!('U', A_list[1]) #TODO Allocation is here

    for i = 2:N
        mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[i-1], B_list[i-1])
        mygemm!('T', 'N', -1.0, B_list[i-1], B_list[i-1], 1.0, A_list[i])
        mypotrf!('U', A_list[i]) #TODO Allocation is here
    end

end

function cholesky_solve!(A_list, B_list, d_list::M, N) where {T, M<:AbstractVector{<:AbstractMatrix{T}}}

    mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[1], d_list[1]);

    for i = 2:N
        mygemm!('T', 'N', -1.0, B_list[i-1], d_list[i-1], 1.0, d_list[i])
        mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[i], d_list[i])
    end

    mytrsm!('L', 'U', 'N', 'N', 1.0, A_list[N], d_list[N])

    for i = N-1:-1:1
        mygemm!('N', 'N', -1.0, B_list[i], d_list[i+1], 1.0, d_list[i])
        mytrsm!('L', 'U', 'N', 'N', 1.0, A_list[i], d_list[i])
    end

end