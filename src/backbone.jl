function cholesky_factorize!(A_list, B_list, N)

    mypotrf!('U', A_list[1]) #TODO Allocation is here

    for i = 2:N

        mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[i-1], B_list[i-1])
        mygemm!('T', 'N', -1.0, B_list[i-1], B_list[i-1], 1.0, A_list[i])
        mypotrf!('U', A_list[i]) #TODO Allocation is here

    end

end

function cholesky_solve!(M_chol_A_list, M_chol_B_list, d_list::M, N) where {T, M<:AbstractVector{<:AbstractMatrix{T}}}

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

function copy_vector_of_arrays!(dest::AbstractVector{<:AbstractArray}, src::AbstractVector{<:AbstractArray})
    @assert length(dest) == length(src) "Vectors must have the same length"

    for i in eachindex(dest, src)
        dest[i] .= src[i]
    end
end

function func2!(factor_list::Vector{<:AbstractMatrix}, d_list_non_separator, temp_list, RHS, P, m, n)

    for i = 1:P-1 #TODO parallelize
        for j = 1:m
            mygemm!('T', 'N', -1.0, factor_list[(i-1)*m+j], d_list_non_separator[(i-1)*m+j], 1.0, temp_list[i])
        end
        RHS[i] .+= view(temp_list[i], 1:n, :)
        RHS[i+1] .+= view(temp_list[i], n+1:2*n, :)
    end

end

function func2!(factor_list::Vector{<:CuMatrix}, d_list_non_separator, temp_list, RHS, P, m, n)

    for i = 1:m
        gemm_batched!('T', 'N', -1.0, factor_list[i:m:end], d_list_non_separator[i:m:end], 1.0, temp_list)
    end
    
    for i = 1:P-1
        RHS[i] .+= view(temp_list[i], 1:n, :)
        RHS[i+1] .+= view(temp_list[i], n+1:2*n, :)
    end
end

function func3!(B_list::Vector{<:AbstractMatrix}, x, d_list, I_separator, P)

    @inbounds for j = 1:P-1 #TODO parallelize
        mygemm!('T', 'N', -1.0, B_list[I_separator[j]], x[I_separator[j]], 1.0, d_list[I_separator[j]+1])
        mygemm!('N', 'N', -1.0, B_list[I_separator[j+1]-1], x[I_separator[j+1]], 1.0, d_list[I_separator[j+1]-1])
    end

end

function func3!(B_list::Vector{<:CuMatrix}, x, d_list, I_separator, P)

    gemm_batched!('T', 'N', -1.0, B_list[I_separator[1:P-1]], x[I_separator[1:P-1]], 1.0, d_list[I_separator[1:P-1].+1])
    gemm_batched!('N', 'N', -1.0, B_list[I_separator[2:P].-1], x[I_separator[2:P]], 1.0, d_list[I_separator[2:P].-1])

end

function func4!(MA_chol_A_list::Vector{<:AbstractMatrix}, MA_chol_B_list, d_list, I_separator, P, m)

    for i = 1:P-1
        cholesky_solve!(MA_chol_A_list[(i-1)*m+1:i*m], MA_chol_B_list[(i-1)*(m-1)+1:i*(m-1)], d_list[(i-1)*m+1:i*m], m)
    end

end

function func4!(MA_chol_A_list::Vector{<:CuMatrix}, MA_chol_B_list, d_list, I_separator, P, m)

    cholesky_solve_batched!(MA_chol_A_list, MA_chol_B_list, d_list, m)

end

function cholesky_solve_batched!(MA_chol_A_list, MA_chol_B_list, d_list, m)

    trsm_batched!('L', 'U', 'T', 'N', 1.0, MA_chol_A_list[1:m:end], d_list[1:m:end]);

    for i = 2:m

        gemm_batched!('T', 'N', -1.0, MA_chol_B_list[i-1:m-1:end], d_list[i-1:m:end], 1.0, d_list[i:m:end])
        trsm_batched!('L', 'U', 'T', 'N', 1.0, MA_chol_A_list[i:m:end], d_list[i:m:end])

    end

    trsm_batched!('L', 'U', 'N', 'N', 1.0, MA_chol_A_list[m:m:end], d_list[m:m:end])

    for i = m-1:-1:1

        gemm_batched!('N', 'N', -1.0, MA_chol_B_list[i:m-1:end], d_list[i+1:m:end], 1.0, d_list[i:m:end])
        trsm_batched!('L', 'U', 'N', 'N', 1.0, MA_chol_A_list[i:m:end], d_list[i:m:end])

    end

end