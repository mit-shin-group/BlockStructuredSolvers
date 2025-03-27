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

function cholesky_factorize_batched!(A_list, B_list, I_separator, P, m)

    I_list = I_separator[1:P-1]

    potrfBatched!('U', A_list[I_list.+1])

    for i = 2:m

        trsm_batched!('L', 'U', 'T', 'N', 1.0, A_list[I_list.+(i-1)], B_list[I_list.+(i-1)])
        gemm_batched!('T', 'N', -1.0, B_list[I_list.+(i-1)], B_list[I_list.+(i-1)], 1.0, A_list[I_list.+i])
        potrfBatched!('U', A_list[I_list.+i])

    end

end

function cholesky_solve_batched!(A_list, B_list, d_list, I_separator, P, m)

    I_list = I_separator[1:P-1]

    trsm_batched!('L', 'U', 'T', 'N', 1.0, A_list[I_list.+1], d_list[1:m:end]);

    for i = 2:m
        gemm_batched!('T', 'N', -1.0, B_list[I_list.+(i-1)], d_list[i-1:m:end], 1.0, d_list[i:m:end])
        trsm_batched!('L', 'U', 'T', 'N', 1.0, A_list[I_list.+i], d_list[i:m:end])
    end

    trsm_batched!('L', 'U', 'N', 'N', 1.0, A_list[I_list.+m], d_list[m:m:end])

    for i = m-1:-1:1
        gemm_batched!('N', 'N', -1.0, B_list[I_list.+(i)], d_list[i+1:m:end], 1.0, d_list[i:m:end])
        trsm_batched!('L', 'U', 'N', 'N', 1.0, A_list[I_list.+(i)], d_list[i:m:end])
    end

end

function copy_vector_of_arrays!(dest::AbstractVector{<:AbstractArray}, src::AbstractVector{<:AbstractArray})
    @assert length(dest) == length(src) "Vectors must have the same length"

    for i in eachindex(dest, src)
        dest[i] .= src[i]
    end
end

#TODO improve how we compute the Schur complement
function compute_schur_complement!(A_list::Vector{<:AbstractMatrix}, B_list::Vector{<:AbstractMatrix}, LHS_A_list, LHS_B_list, temp_B_list, factor_list, factor_list_temp, M_2n_list, I_separator, P, m, n)

    @inbounds for i = 1:P-1

        # Perform Cholesky factorization
        cholesky_factorize!(A_list[I_separator[i]+1:I_separator[i]+m], B_list[I_separator[i]+1:I_separator[i]+m-1], m) #tiny allocation here about 2, belongs to potrf!

        # Set up M_n_2n_list_! for Schur complement
        copyto!(view(factor_list_temp[(i-1)*m+1], :, 1:n), temp_B_list[2*(i-1)+1]')
        copyto!(view(factor_list_temp[i*m], :, n+1:2*n), temp_B_list[2*i])
    end

    # Copy factor_list_temp to M_n_2n_list_2
    copy_vector_of_arrays!(factor_list, factor_list_temp)

    @inbounds for i = 1:P-1
        # Solve using Cholesky factors
        cholesky_solve!(A_list[I_separator[i]+1:I_separator[i]+m], B_list[I_separator[i]+1:I_separator[i]+m-1], factor_list[(i-1)*m+1:i*m], m)

        # Compute Schur complement
        for j = 1:m
            mygemm!('T', 'N', 1.0, factor_list_temp[(i-1)*m+j], factor_list[(i-1)*m+j], 1.0, M_2n_list[i])
        end

        # Update LHS matrices
        LHS_A_list[i] .-= view(M_2n_list[i], 1:n, 1:n)
        LHS_A_list[i+1] .-= view(M_2n_list[i], n+1:2*n, n+1:2*n)
        LHS_B_list[i] .-= view(M_2n_list[i], 1:n, n+1:2*n)

    end

end

function compute_schur_complement!(A_list::Vector{<:CuMatrix}, B_list::Vector{<:CuMatrix}, LHS_A_list, LHS_B_list, temp_B_list, factor_list, factor_list_temp, M_2n_list, I_separator, P, m, n)
           
    # Perform Cholesky factorization
    cholesky_factorize_batched!(A_list, B_list, I_separator, P, m)
    
    @inbounds for i = 1:P-1
        # Set up M_n_2n_list_! for Schur complement
        copyto!(view(factor_list_temp[(i-1)*m+1], :, 1:n), temp_B_list[2*(i-1)+1]')
        copyto!(view(factor_list_temp[i*m], :, n+1:2*n), temp_B_list[2*i])
    end

    # Copy factor_list_temp to M_n_2n_list_2
    copy_vector_of_arrays!(factor_list, factor_list_temp)

    # Solve using Cholesky factors
    cholesky_solve_batched!(A_list, B_list, factor_list, I_separator, P, m)

    # Compute Schur complement
    for j = 1:m
        gemm_batched!('T', 'N', 1.0, factor_list_temp[j:m:end], factor_list[j:m:end], 1.0, M_2n_list)
    end

    @inbounds for i = 1:P-1
        # Update LHS matrices
        LHS_A_list[i] .-= view(M_2n_list[i], 1:n, 1:n)
        LHS_A_list[i+1] .-= view(M_2n_list[i], n+1:2*n, n+1:2*n)
        LHS_B_list[i] .-= view(M_2n_list[i], 1:n, n+1:2*n)

    end

end

function compute_schur_rhs!(factor_list::Vector{<:AbstractMatrix}, d_list_non_separator, temp_list, RHS, P, m, n)

    for i = 1:P-1
        for j = 1:m
            mygemm!('T', 'N', -1.0, factor_list[(i-1)*m+j], d_list_non_separator[(i-1)*m+j], 1.0, temp_list[i])
        end
        RHS[i] .+= view(temp_list[i], 1:n, :)
        RHS[i+1] .+= view(temp_list[i], n+1:2*n, :)
    end

end

function compute_schur_rhs!(factor_list::Vector{<:CuMatrix}, d_list_non_separator, temp_list, RHS, P, m, n)

    for i = 1:m
        gemm_batched!('T', 'N', -1.0, factor_list[i:m:end], d_list_non_separator[i:m:end], 1.0, temp_list)
    end
    
    for i = 1:P-1
        RHS[i] .+= view(temp_list[i], 1:n, :)
        RHS[i+1] .+= view(temp_list[i], n+1:2*n, :)
    end
end

function update_boundary_solution!(temp_B_list::Vector{<:AbstractMatrix}, x, d_list, I_separator, P)

    @inbounds for j = 1:P-1
        mygemm!('T', 'N', -1.0, temp_B_list[2*(j-1)+1], x[I_separator[j]], 1.0, d_list[I_separator[j]+1])
        mygemm!('N', 'N', -1.0, temp_B_list[2*j], x[I_separator[j+1]], 1.0, d_list[I_separator[j+1]-1])
    end

end

function update_boundary_solution!(temp_B_list::Vector{<:CuMatrix}, x, d_list, I_separator, P)

    gemm_batched!('T', 'N', -1.0, temp_B_list[1:2:end], x[I_separator[1:P-1]], 1.0, d_list[I_separator[1:P-1].+1])
    gemm_batched!('N', 'N', -1.0, temp_B_list[2:2:end], x[I_separator[2:P]], 1.0, d_list[I_separator[2:P].-1])

end

function solve_non_separator_blocks!(A_list::Vector{<:AbstractMatrix}, B_list::Vector{<:AbstractMatrix}, d_list, I_separator, P, m)

    for i = 1:P-1
        cholesky_solve!(A_list[I_separator[i]+1:I_separator[i]+m], B_list[I_separator[i]+1:I_separator[i]+m-1], d_list[(i-1)*m+1:i*m], m)
    end

end

function solve_non_separator_blocks!(A_list::Vector{<:CuMatrix}, B_list::Vector{<:CuMatrix}, d_list, I_separator, P, m)

    cholesky_solve_batched!(A_list, B_list, d_list, I_separator, P, m)

end