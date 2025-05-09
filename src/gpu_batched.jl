struct BlockTriDiagData_batched{
    T, 
    MT <: AbstractArray{T},
    VMT <: Vector{MT}
    }

    N::Int
    m::Int
    n::Int
    P::Int

    A_vec::MT
    A_tensor::MT
    A_list::VMT
    A_ptrs::CuVector{CuPtr{T}}

    B_vec::MT
    B_tensor::MT
    B_list::VMT
    B_ptrs::CuVector{CuPtr{T}}

    d_vec::MT
    d_tensor::MT
    d_list::VMT
    d_ptrs::CuVector{CuPtr{T}}

    LHS_A_vec::MT
    LHS_A_tensor::MT
    LHS_A_list::VMT
    LHS_A_ptrs::CuVector{CuPtr{T}}

    LHS_B_vec::MT
    LHS_B_tensor::MT
    LHS_B_list::VMT
    LHS_B_ptrs::CuVector{CuPtr{T}}

    M_vec_1::MT
    M_tensor_1::MT
    M_list_1::VMT
    M_ptrs_1::CuVector{CuPtr{T}}

    M_vec_2::MT
    M_tensor_2::MT  
    M_list_2::VMT
    M_ptrs_2::CuVector{CuPtr{T}}

    M_2n_vec::MT
    M_2n_tensor::MT
    M_2n_list::VMT
    M_2n_ptrs::CuVector{CuPtr{T}}

    F_vec::MT
    F_tensor::MT
    F_list::VMT
    F_ptrs::CuVector{CuPtr{T}}

    G_vec::MT
    G_tensor::MT
    G_list::VMT
    G_ptrs::CuVector{CuPtr{T}}

    b_vec::MT
    b_tensor::MT    
    b_list::VMT
    b_ptrs::CuVector{CuPtr{T}}

    NextData::Union{BlockTriDiagData_batched, BlockTriDiagData_seq}

end

function create_data(N::Int, m::Int, n::Int, P::Int, next_Data::Union{BlockTriDiagData_batched, BlockTriDiagData_seq}, T::Type{<:Real}=Float64)

    MT = CuArray{T}

    A_vec, A_tensor, A_list, A_ptrs = create_matrix_list(N, n, n, MT)
    B_vec, B_tensor, B_list, B_ptrs = create_matrix_list(N-1, n, n, MT)

    d_vec, d_tensor, d_list, d_ptrs = create_matrix_list(N, n, 1, MT)    
    
    LHS_A_vec, LHS_A_tensor, LHS_A_list, LHS_A_ptrs = create_matrix_list(P, n, n, MT)
    LHS_B_vec, LHS_B_tensor, LHS_B_list, LHS_B_ptrs = create_matrix_list(P-1, n, n, MT)

    M_vec_1, M_tensor_1, M_list_1, M_ptrs_1 = create_matrix_list(P-1, n, n, MT)
    M_vec_2, M_tensor_2, M_list_2, M_ptrs_2 = create_matrix_list(P-1, n, n, MT)

    M_2n_vec, M_2n_tensor, M_2n_list, M_2n_ptrs = create_matrix_list(P-1, 2*n, 2*n, MT)

    F_vec, F_tensor, F_list, F_ptrs = create_matrix_list(N, n, 2*n, MT)
    G_vec, G_tensor, G_list, G_ptrs = create_matrix_list(N, n, 2*n, MT)

    b_vec, b_tensor, b_list, b_ptrs = create_matrix_list(P-1, 2*n, 1, MT)

    data = BlockTriDiagData_batched(
        N, m, n, P,
        A_vec, A_tensor, A_list, A_ptrs,
        B_vec, B_tensor, B_list, B_ptrs,
        d_vec, d_tensor, d_list, d_ptrs,
        LHS_A_vec, LHS_A_tensor, LHS_A_list, LHS_A_ptrs,
        LHS_B_vec, LHS_B_tensor, LHS_B_list, LHS_B_ptrs,
        M_vec_1, M_tensor_1, M_list_1, M_ptrs_1,
        M_vec_2, M_tensor_2, M_list_2, M_ptrs_2,
        M_2n_vec, M_2n_tensor, M_2n_list, M_2n_ptrs,
        F_vec, F_tensor, F_list, F_ptrs,
        G_vec, G_tensor, G_list, G_ptrs,
        b_vec, b_tensor, b_list, b_ptrs,
        next_Data
    )

    return data
end

# function compute_partition(N::Int)

#     m = 4
#     P = length(1:(m+1):N)

#     N_list = [N]
#     m_list = [m]
#     P_list = [P]

#     return N_list, m_list, P_list

# end

############  core backwards‑builder (unchanged)  ############################
function _build_sequence(start::Int)
    P_list = Int[]; N_list = Int[]; m_list = Int[]
    Ncur   = start
    while true
        best_m = best_P = backup_m = backup_P = nothing
        max_m  = (Ncur - 3) ÷ 2                 # ensure P_prev ≥ 3
        for m in 3:max_m
            num = Ncur + m                      # P_prev = (Ncur + m)/(m+1)
            den = m + 1
            if num % den == 0
                Pprev = num ÷ den
                if Pprev ≥ 3
                    if m < Pprev
                        best_m, best_P = m, Pprev        # preferred
                        break
                    elseif backup_m === nothing
                        backup_m, backup_P = m, Pprev    # acceptable
                    end
                end
            end
        end
        if best_m === nothing && backup_m !== nothing
            best_m, best_P = backup_m, backup_P
        end
        best_m === nothing && break                     # no predecessor
        pushfirst!(P_list, best_P)
        pushfirst!(N_list, Ncur)
        pushfirst!(m_list, best_m)
        Ncur = best_P
    end
    return P_list, N_list, m_list
end

############  public API: “nearest larger N”  ################################
"""
    find_sequence_upper(N::Integer) -> (N_use, P_list, N_list, m_list)

Return the *smallest* integer `N_use ≥ N` for which a valid sequence exists,
together with the three lists that satisfy  
`N_i = (P_i - 1) * m_i + P_i` and `P_{i+1} = N_i`.

If the original `N` already works it is returned unchanged.
"""
function find_sequence_upper(N::Int)
    N_use = N
    while true
        P_list, N_list, m_list = _build_sequence(N_use)
        !isempty(P_list) && return P_list, N_list, m_list
        N_use += 1
    end
end


function initialize_batched(N::Int, n::Int)

    P_list, N_list, m_list = find_sequence_upper(N)

    N = P_list[1]

    data = initialize_seq(N, n)

    for (N, m, P) in zip(N_list, m_list, P_list)

        data = create_data(N, m, n, P, data)

    end

    return data
    
end

function factorize!(data::BlockTriDiagData_batched)

    P = data.P
    m = data.m
    n = data.n
    N = data.N

    A_tensor = data.A_tensor
    B_tensor = data.B_tensor
    M_tensor_1 = data.M_tensor_1
    M_tensor_2 = data.M_tensor_2
    LHS_A_tensor = data.LHS_A_tensor
    LHS_B_tensor = data.LHS_B_tensor
    G_tensor = data.G_tensor
    F_tensor = data.F_tensor
    M_2n_tensor = data.M_2n_tensor

    A_ptrs = data.A_ptrs
    B_ptrs = data.B_ptrs
    F_ptrs = data.F_ptrs
    G_ptrs = data.G_ptrs
    M_2n_ptrs = data.M_2n_ptrs
    LHS_A_ptrs = data.LHS_A_ptrs
    LHS_B_ptrs = data.LHS_B_ptrs

    T = eltype(A_tensor)

    copyto!(M_tensor_1, view(B_tensor, :, :, 1:(m+1):(N-1)));
    copyto!(M_tensor_2, view(B_tensor, :, :, (m+1):(m+1):(N-1)));
    copyto!(LHS_A_tensor, view(A_tensor, :, :, 1:(m+1):N));
    copyto!(view(G_tensor, :, 1:n, 2:(m+1):N), permutedims(M_tensor_1, (2, 1, 3)));
    copyto!(view(G_tensor, :, n+1:2*n, (m+1):(m+1):N), M_tensor_2);
    copyto!(F_tensor, G_tensor);

    cholesky_factorize_batched!(A_ptrs, B_ptrs, P-1, m+1, 2, m, n)
    cholesky_solve_batched!(A_ptrs, B_ptrs, F_ptrs, P-1, m+1, 2, m, n, 2*n)

    for j = 2:m+1
        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            2*n, 2*n, n, one(T),
            G_ptrs[j:(m+1):end], n, F_ptrs[j:(m+1):end], n,
            one(T), M_2n_ptrs, 2*n, P-1)
    end

    view(LHS_A_tensor, :, :, 1:P-1) .-= view(M_2n_tensor, 1:n, 1:n, :);
    view(LHS_A_tensor, :, :, 2:P) .-= view(M_2n_tensor, n+1:2*n, n+1:2*n, :);
    LHS_B_tensor .-= view(M_2n_tensor, 1:n, n+1:2*n, :);

    copyto!(data.NextData.A_tensor, LHS_A_tensor)
    copyto!(data.NextData.B_tensor, LHS_B_tensor)
    factorize!(data.NextData)
    
end

function solve!(data::BlockTriDiagData_batched)

    P = data.P
    m = data.m
    n = data.n
    N = data.N

    F_ptrs = data.F_ptrs
    b_ptrs = data.b_ptrs
    d_ptrs = data.d_ptrs
    A_ptrs = data.A_ptrs
    B_ptrs = data.B_ptrs
    LHS_A_ptrs = data.LHS_A_ptrs
    LHS_B_ptrs = data.LHS_B_ptrs

    M_ptrs_1 = data.M_ptrs_1
    M_ptrs_2 = data.M_ptrs_2
    d_tensor = data.d_tensor
    b_tensor = data.b_tensor

    T = eltype(b_tensor)
    
    for i = 2:m+1
        CUBLAS.cublasDgemmBatched(
            CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
            2*n, 1, n, -one(T),
            F_ptrs[i:(m+1):end], n, d_ptrs[i:(m+1):end], n,
            one(T), b_ptrs, 2*n, P-1)
    end
    
    view(d_tensor, :, :, 1:(m+1):N-1) .+= view(b_tensor, 1:n, :, :);
    view(d_tensor, :, :, (m+2):(m+1):N) .+= view(b_tensor, n+1:2*n, :, :);
    
    copyto!(data.NextData.d_tensor, view(d_tensor, :, :, 1:(m+1):N))
    solve!(data.NextData)
    copyto!(view(d_tensor, :, :, 1:(m+1):N), data.NextData.d_tensor)

    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_T, CUBLAS.CUBLAS_OP_N,
        n, 1, n, -one(T),
        M_ptrs_1, n, d_ptrs[1:(m+1):end-1], n,
        one(T), d_ptrs[2:(m+1):end], n, P-1)
    CUBLAS.cublasDgemmBatched(
        CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N,
        n, 1, n, -one(T),
        M_ptrs_2, n, d_ptrs[(m+2):(m+1):end], n,
        one(T), d_ptrs[(m+1):(m+1):end], n, P-1)
    
    cholesky_solve_batched!(A_ptrs, B_ptrs, d_ptrs, P-1, m+1, 2, m, n, 1)
    
end