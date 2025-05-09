struct BlockTriDiagData_seq{
    T, 
    MT2 <: CuArray{T, 2},
    MT3 <: CuArray{T, 3},
    VMT <: Vector{MT2},
    VPtr <: CuVector{CuPtr{T}}
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

function create_matrix_list(N::Int, n1::Int, n2::Int, T)

    M_vec = CuArray{T, 2}(zeros(N*n1*n2, 1))
    M_tensor = unsafe_wrap(CuArray{T, 3}, pointer(M_vec), (n1, n2, N); own=false)
    M_list = Vector{CuMatrix{T}}(undef, N);
    ptr = pointer(M_tensor)

    for i in 1:N
        M_list[i] = unsafe_wrap(CuMatrix{T}, ptr + n1*n2*(i-1)*sizeof(T), (n1, n2); own=false)
    end

    M_ptrs = CUBLAS.unsafe_batch(M_list)

    return M_vec, M_tensor, M_list, M_ptrs
end

function initialize_seq(N::Int, n::Int, T::Type{<:Real}=Float64)

    A_vec, A_tensor, A_list, A_ptrs = create_matrix_list(N, n, n, T)
    B_vec, B_tensor, B_list, B_ptrs = create_matrix_list(N-1, n, n, T)
    d_vec, d_tensor, d_list, d_ptrs = create_matrix_list(N, n, 1, T)

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

    @CUDA.allowscalar cholesky_factorize!(A_ptrs, B_ptrs, N, n)

end

function solve!(data::BlockTriDiagData_seq)

    N = data.N
    n = data.n

    A_ptrs = data.A_ptrs
    B_ptrs = data.B_ptrs
    d_ptrs = data.d_ptrs

    @CUDA.allowscalar cholesky_solve!(A_ptrs, B_ptrs, d_ptrs, N, n, 1)

end