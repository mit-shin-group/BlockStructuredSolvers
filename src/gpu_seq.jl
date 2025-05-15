struct BlockTriDiagData_seq{
    T, 
    MT2 <: AbstractArray{T, 2},
    MT3, # <: CuArray{T, 3},
    VMT, # <: Vector{MT2},
    VPtr # <: CuVector{CuPtr{T}}
    }

    N::Int
    n::Int

    A_vec::MT2
    A_tensor::MT3
    A_list::VMT
    A_ptrs::VPtr

    B_vec::MT2
    B_tensor::MT3
    B_list::VMT
    B_ptrs::VPtr

    d_vec::MT2
    d_tensor::MT3
    d_list::VMT
    d_ptrs::VPtr

end

function create_matrix_list(N::Int, n1::Int, n2::Int, ::Type{T}, ::Type{M}) where {T, M}

    amd = M.body.body.body.name.name == :ROCArray ? true : false
    M_vec = M{T, 2}(zeros(N*n1*n2, 1))
    if amd
        M_tensor = unsafe_wrap(M{T, 3}, pointer(M_vec), (n1, n2, N); lock=false)
    else
        M_tensor = unsafe_wrap(M{T, 3}, pointer(M_vec), (n1, n2, N))
    end
    M_list = Vector{M{T, 2}}(undef, N);
    ptr = pointer(M_tensor)

    for i in 1:N
        if amd
            M_list[i] = unsafe_wrap(M{T, 2}, ptr + n1*n2*(i-1)*sizeof(T), (n1, n2), lock=false)
        else
            M_list[i] = unsafe_wrap(M{T, 2}, ptr + n1*n2*(i-1)*sizeof(T), (n1, n2))
        end
    end

    M_ptrs = device_batch(M_list)

    return M_vec, M_tensor, M_list, M_ptrs
end

function initialize_seq(N::Int, n::Int, ::Type{T}, ::Type{M}) where {T, M}

    A_vec, A_tensor, A_list, A_ptrs = create_matrix_list(N, n, n, T, M)
    B_vec, B_tensor, B_list, B_ptrs = create_matrix_list(N-1, n, n, T, M)
    d_vec, d_tensor, d_list, d_ptrs = create_matrix_list(N, n, 1, T, M)

    data = BlockTriDiagData_seq(
        N, n,
        A_vec, A_tensor, A_list, A_ptrs,
        B_vec, B_tensor, B_list, B_ptrs,
        d_vec, d_tensor, d_list, d_ptrs
    )

    return data
end

function factorize!(data::BlockTriDiagData_seq)

    N = data.N
    n = data.n

    A_ptrs = data.A_ptrs
    B_ptrs = data.B_ptrs

    @allowscalar cholesky_factorize!(A_ptrs, B_ptrs, N, n) #TODO check if works for both CUDA and ROCm

end

function solve!(data::BlockTriDiagData_seq)

    N = data.N
    n = data.n

    A_ptrs = data.A_ptrs
    B_ptrs = data.B_ptrs
    d_ptrs = data.d_ptrs

    @allowscalar cholesky_solve!(A_ptrs, B_ptrs, d_ptrs, N, n, 1) #TODO check if works for both CUDA and ROCm

end