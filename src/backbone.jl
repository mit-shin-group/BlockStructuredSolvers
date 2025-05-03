function cholesky_factorize!(A_list, B_list, N)

    mypotrf!('U', A_list[1]) #TODO Allocation is here

    for i = 2:N
        mytrsm!('L', 'U', 'T', 'N', 1.0, A_list[i-1], B_list[i-1])
        mygemm!('T', 'N', -1.0, B_list[i-1], B_list[i-1], 1.0, A_list[i])
        mypotrf!('U', A_list[i]) #TODO Allocation is here
    end

end

function cholesky_factorize!(A_ptrs::CuVector{<:CuPtr{T}}, B_ptrs::CuVector{<:CuPtr{T}}, N, n) where {T<:Union{Float32,Float64}}

    CUSOLVER.cusolverDnDpotrf('U', A_list[1]) #TODO Allocation is here

    for i = 2:N
        CUBLAS.cublasDtrsm_v2('L', 'U', 'T', 'N', 1.0, A_list[i-1], B_list[i-1])
        CUBLAS.cublasDgemm_v2('T', 'N', -1.0, B_list[i-1], B_list[i-1], 1.0, A_list[i])
        CUSOLVER.cusolverDnDpotrf('U', A_list[i]) #TODO Allocation is here
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

function cholesky_factorize_batched!(A_ptrs::CuVector{<:CuPtr{T}}, B_ptrs::CuVector{<:CuPtr{T}}, P, m, n) where {T<:Union{Float32,Float64}}

    # ---- one info buffer and one cuSOLVER handle ---------------------------
    bsz     = length(A_ptrs[1:m:end])
    info_array  = CUDA.zeros(Int32, bsz)

    # ---- first block row ----------------------------------------------------
    CUSOLVER.cusolverDnDpotrfBatched(
        CUSOLVER.dense_handle(), CUBLAS.CUBLAS_FILL_MODE_UPPER,
        n, A_ptrs[1:m:end], n, pointer(info_array), bsz)
    
    # ---- remaining block rows ----------------------------------------------
    for i in 2:m-1

        CUBLAS.cublasDtrsmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
            CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
            n, n, one(T), A_ptrs[i-1:m:end], n, B_ptrs[(i-1):m:end], n, bsz)

        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            n, n, n, -one(T),
            B_ptrs[(i-1):m:end], n, B_ptrs[(i-1):m:end], n,
            one(T), A_ptrs[i:m:end], n, bsz)

        CUSOLVER.cusolverDnDpotrfBatched(
            CUSOLVER.dense_handle(), CUBLAS.CUBLAS_FILL_MODE_UPPER,
            n, A_ptrs[i:m:end], n, pointer(info_array), bsz)

    end
    
end

function cholesky_solve_batched!(A_ptrs::CuVector{<:CuPtr{T}}, B_ptrs::CuVector{<:CuPtr{T}}, d_ptrs::CuVector{<:CuPtr{T}}, P, m, n) where {T<:Union{Float32,Float64}}

    bsz = length(A_ptrs[1:m:end])
    nd  = size(d_ptrs[1], 2)

    CUBLAS.cublasDtrsmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
        CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
        n, nd, one(T), A_ptrs[1:m:end], n, d_ptrs[1:m:end], n, bsz)


    for i = 2:m-1
        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            n, nd, n, -one(T),
            B_ptrs[(i-1):m:end], n, d_ptrs[(i-1):m:end], n,
            one(T), d_ptrs[i:m:end], n, bsz)

        CUBLAS.cublasDtrsmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
            CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_DIAG_NON_UNIT,
            n, nd, one(T), A_ptrs[i:m:end], n, d_ptrs[i:m:end], n, bsz)
    end

    for i = m-1:-1:2
        CUBLAS.cublasDtrsmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
            CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT,
            n, nd, one(T), A_ptrs[i:m:end], n, d_ptrs[i:m:end], n, bsz)

        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
            n, nd, n, -one(T),
            B_ptrs[(i-1):m:end], n, d_ptrs[i:m:end], n,
            one(T), d_ptrs[(i-1):m:end], n, bsz)
    end

    CUBLAS.cublasDtrsmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_SIDE_LEFT, CUBLAS.CUBLAS_FILL_MODE_UPPER,
        CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_DIAG_NON_UNIT,
        n, nd, one(T), A_ptrs[1:m:end], n, d_ptrs[1:m:end], n, bsz)    

end

function copy_vector_of_arrays!(dest::AbstractVector{<:AbstractArray}, src::AbstractVector{<:AbstractArray})
    @assert length(dest) == length(src) "Vectors must have the same length"

    for i in eachindex(dest, src)
        dest[i] .= src[i]
    end
end

function add_vector_of_arrays!(dest::AbstractVector{<:AbstractArray}, src::AbstractVector{<:AbstractArray})
    @assert length(dest) == length(src) "Vectors must have the same length"

    for i in eachindex(dest, src)
        dest[i] .+= src[i]
    end
end

#TODO improve how we compute the Schur complement
function compute_schur_complement!(A_list::Vector{<:AbstractMatrix}, B_list::Vector{<:AbstractMatrix}, LHS_A_list, LHS_B_list, temp_B_list, factor_list, factor_list_temp, M_2n_list, I_separator, P, m, n)

    @inbounds for i = 1:P-1

        # Perform Cholesky factorization
        cholesky_factorize!(view(A_list, I_separator[i]+1:I_separator[i+1]-1), view(B_list, I_separator[i]+1:I_separator[i+1]-2), I_separator[i+1] - I_separator[i]-1) #tiny allocation here about 2, belongs to potrf!

        # Set up M_n_2n_list_! for Schur complement
        copyto!(view(factor_list_temp[I_separator[i]+1], :, 1:n), temp_B_list[2*(i-1)+1]')
        copyto!(view(factor_list_temp[I_separator[i+1]-1], :, n+1:2*n), temp_B_list[2*i])
    end

    # Copy factor_list_temp to M_n_2n_list_2
    copy_vector_of_arrays!(factor_list, factor_list_temp)

    @inbounds for i = 1:P-1
        # Solve using Cholesky factors
        cholesky_solve!(view(A_list, I_separator[i]+1:I_separator[i+1]-1), view(B_list, I_separator[i]+1:I_separator[i+1]-2), view(factor_list, I_separator[i]+1:I_separator[i+1]-1), I_separator[i+1] - I_separator[i]-1)

        # Compute Schur complement
        for j = I_separator[i]+1:I_separator[i+1]-1
            mygemm!('T', 'N', 1.0, factor_list_temp[j], factor_list[j], 1.0, M_2n_list[i])
        end

        # Update LHS matrices
        LHS_A_list[i] .-= view(M_2n_list[i], 1:n, 1:n)
        LHS_A_list[i+1] .-= view(M_2n_list[i], n+1:2*n, n+1:2*n)
        LHS_B_list[i] .-= view(M_2n_list[i], 1:n, n+1:2*n)

    end

end

function compute_schur_complement!(A_ptrs::CuVector{<:CuPtr{T}}, B_ptrs::CuVector{<:CuPtr{T}}, factor_ptrs::CuVector{<:CuPtr{T}}, I_separator, P, m, n) where {T<:Union{Float32,Float64}}
    
    bsz = length(A_ptrs[2:m+1:end])

    # Perform Cholesky factorization
    cholesky_factorize_batched!(A_ptrs[2:I_separator[P]], B_ptrs[2:I_separator[P]-1], P, m, n)
    
    # cholesky_factorize!(view(A_list, I_separator[P-1]+1:I_separator[P]), view(B_list, I_separator[P-1]+1:I_separator[P]-1), I_separator[P]-I_separator[P-1]-1)


    # Copy factor_list_temp to M_n_2n_list_2

    # Solve using Cholesky factors
    # cholesky_solve_batched!(view(A_list, 2:I_separator[P-1]), view(B_list, 2:I_separator[P-1]), view(factor_list, 2:I_separator[P-1]), m+1)
    cholesky_solve_batched!(A_ptrs[2:I_separator[P]], A_ptrs[2:I_separator[P]-1], factor_list[2:I_separator[P]], m+1)

    # last partition
    # cholesky_solve!(view(A_list, I_separator[P-1]+1:I_separator[P]), view(B_list, I_separator[P-1]+1:I_separator[P]-1), view(factor_list, I_separator[P-1]+1:I_separator[P]), I_separator[P]-I_separator[P-1]-1)

    # Compute Schur complement
    factor_ptrs = CUBLAS.unsafe_batch(factor_list)
    factor_ptrs_temp = CUBLAS.unsafe_batch(factor_list_temp)
    M_2n_ptrs = CUBLAS.unsafe_batch(M_2n_list)

    # println(size(factor_list_temp[2]))
    # println(size(factor_list[2]))
    # gemm_batched!('T', 'N', 1.0, factor_list_temp[2:(m+1):end], factor_list[2:(m+1):end], 0.0, M_2n_list)
    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
        2*n, 2*n, n, one(T),
        factor_ptrs_temp[2:(m+1):I_separator[P]-1], n, factor_ptrs[2:(m+1):I_separator[P]-1], n,
        zero(T), M_2n_ptrs, 2*n, bsz)

    for j = 3:m+1
        # gemm_batched!('T', 'N', 1.0, factor_list_temp[j:(m+1):end], factor_list[j:(m+1):end], 1.0, M_2n_list)
        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            2*n, 2*n, n, one(T),
            factor_ptrs_temp[j:(m+1):I_separator[P]-1], n, factor_ptrs[j:(m+1):I_separator[P]-1], n,
            one(T), M_2n_ptrs, 2*n, bsz)
    end

    # last partition
    # for j = I_separator[P-1]+1:I_separator[P]-1
    #     mygemm!('T', 'N', 1.0, factor_list_temp[j], factor_list[j], 1.0, M_2n_list[P-1])
    # end


end

function compute_schur_rhs!(factor_list::Vector{<:AbstractMatrix}, d_list, temp_list, I_separator, P, m, n)

    for i = 1:P-1
        fill!(temp_list[i], 0.0)
        for j = I_separator[i]+1:I_separator[i+1]-1
            mygemm!('T', 'N', -1.0, factor_list[j], d_list[j], 1.0, temp_list[i])
        end
        d_list[I_separator[i]] .+= view(temp_list[i], 1:n, :)
        d_list[I_separator[i+1]] .+= view(temp_list[i], n+1:2*n, :)
    end

end

function compute_schur_rhs!(factor_ptrs::CuVector{<:CuPtr{T}}, d_ptrs::CuVector{<:CuPtr{T}}, temp_ptrs::CuVector{<:CuPtr{T}}, P, m, n) where {T<:Union{Float32,Float64}}

    # println(size(factor_list[1]), size(d_list[1]), size(temp_list[1]))
    bsz = length(factor_ptrs[2:m+1:end])

    # gemm_batched!('T', 'N', -1.0, factor_list[2:(m+1):end], d_list[2:(m+1):end], 0.0, temp_list[1:length(factor_list[2:(m+1):end])])
    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
        2*n, 1, n, -one(T),
        factor_ptrs[2:(m+1):end], n, d_ptrs[2:(m+1):end], n,
        zero(T), temp_ptrs, 2*n, bsz)

    for i = 3:m+1
        # gemm_batched!('T', 'N', -1.0, factor_list[i:(m+1):end], d_list[i:(m+1):end], 1.0, temp_list[1:length(factor_list[i:(m+1):end])])
        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            2*n, 1, n, -one(T),
            factor_ptrs[i:(m+1):end], n, d_ptrs[i:(m+1):end], n,
            one(T), temp_ptrs, 2*n, bsz)
    end
    

end

function update_boundary_solution!(temp_B_list::Vector{<:AbstractMatrix}, d_list, I_separator, P)

    @inbounds for j = 1:P-1
        mygemm!('T', 'N', -1.0, temp_B_list[2*(j-1)+1], d_list[I_separator[j]], 1.0, d_list[I_separator[j]+1])
        mygemm!('N', 'N', -1.0, temp_B_list[2*j], d_list[I_separator[j+1]], 1.0, d_list[I_separator[j+1]-1])
    end

end

function update_boundary_solution!(temp_B_ptrs::CuVector{<:CuPtr{T}}, d_ptrs::CuVector{<:CuPtr{T}}, I_separator, P, m, n, nd) where {T<:Union{Float32,Float64}}
    
    bsz = P-1
    # gemm_batched!('T', 'N', -1.0, temp_B_list[1:2:end], d_list[I_separator[1:P-1]], 1.0, d_list[I_separator[1:P-1].+1])
    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
        n, nd, n, -one(T),
        temp_B_ptrs[1:2:end], n, d_ptrs[I_separator[1:P-1]], n,
        one(T), d_ptrs[I_separator[1:P-1].+1], n, bsz)
    # gemm_batched!('N', 'N', -1.0, temp_B_list[2:2:end], d_list[I_separator[2:P]], 1.0, d_list[I_separator[2:P].-1])
    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
        n, nd, n, -one(T),
        temp_B_ptrs[2:2:end], n, d_ptrs[I_separator[2:P]], n,
        one(T), d_ptrs[I_separator[2:P].-1], n, bsz)

end

function solve_non_separator_blocks!(A_list::Vector{<:AbstractMatrix}, B_list::Vector{<:AbstractMatrix}, d_list, I_separator, P, m)

    for i = 1:P-1
        m_temp = I_separator[i+1] - I_separator[i]-1;
        cholesky_solve!(view(A_list, I_separator[i]+1:I_separator[i+1]-1), view(B_list, I_separator[i]+1:I_separator[i+1]-2), view(d_list, I_separator[i]+1:I_separator[i+1]-1), m_temp)
    end

end

function solve_non_separator_blocks!(A_ptrs::CuVector{<:CuPtr{T}}, B_ptrs::CuVector{<:CuPtr{T}}, d_ptrs::CuVector{<:CuPtr{T}}, I_separator, P, m, n) where {T<:Union{Float32,Float64}}

    cholesky_solve_batched!(A_ptrs[2:I_separator[P]], B_ptrs[2:I_separator[P]-1], d_ptrs[2:I_separator[P]], P, m+1, n)

end