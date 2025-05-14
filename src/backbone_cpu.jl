function cholesky_factorize!(A_list::Vector{<:AbstractMatrix{T}}, B_list::Vector{<:AbstractMatrix{T}}, N) where T

    mypotrf!('U', A_list[1]) #TODO Allocation is here

    for i = 2:N
        mytrsm!('L', 'U', 'T', 'N', one(T), A_list[i-1], B_list[i-1])
        mygemm!('T', 'N', -one(T), B_list[i-1], B_list[i-1], one(T), A_list[i])
        mypotrf!('U', A_list[i]) #TODO Allocation is here
    end

end

function cholesky_solve!(A_list::Vector{<:AbstractMatrix{T}}, B_list::Vector{<:AbstractMatrix{T}}, d_list::Vector{<:AbstractMatrix{T}}, N) where T

    mytrsm!('L', 'U', 'T', 'N', one(T), A_list[1], d_list[1]);

    for i = 2:N
        mygemm!('T', 'N', -one(T), B_list[i-1], d_list[i-1], one(T), d_list[i])
        mytrsm!('L', 'U', 'T', 'N', one(T), A_list[i], d_list[i])
    end

    mytrsm!('L', 'U', 'N', 'N', one(T), A_list[N], d_list[N])

    for i = N-1:-1:1
        mygemm!('N', 'N', -one(T), B_list[i], d_list[i+1], one(T), d_list[i])
        mytrsm!('L', 'U', 'N', 'N', one(T), A_list[i], d_list[i])
    end

end