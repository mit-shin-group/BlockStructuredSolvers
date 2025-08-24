using SparseArrays

export construct_block_tridiagonal, generate_data, to_gpu

function generate_data(N::Int, n::Int, T::Type)
    # Generate CPU matrices
    A_list = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        temp = randn(n, n)
        A_list[i] = temp * temp' + n * I
    end
    
    B_list = Vector{Matrix{T}}(undef, N-1)
    for i in 1:N-1
        temp = randn(n, n)
        B_list[i] = temp
    end
    
    x_list = Vector{Matrix{T}}(undef, N)
    x = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        x_list[i] = rand(n, 1)
        x[i] = zeros(n, 1)
    end
    
    d_list = Vector{Matrix{T}}(undef, N)
    d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2]
    @views for i = 2:N-1
        d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
    end
    d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N]
    
    return A_list, B_list, x_list, x, d_list
end


function to_nvidia_gpu(A_list, B_list, x_list, x, d_list)
    T = eltype(A_list[1])
    # Convert CPU arrays to GPU arrays on NVIDIA GPUs
    A_list_gpu = [CuArray{T}(A) for A in A_list]
    B_list_gpu = [CuArray{T}(B) for B in B_list]
    x_list_gpu = [CuArray{T}(x) for x in x_list]
    x_gpu = [CuArray{T}(x) for x in x]
    d_list_gpu = [CuArray{T}(d) for d in d_list]
    return A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu
end

function to_amd_gpu(A_list, B_list, x_list, x, d_list)
    T = eltype(A_list[1])
    # Convert CPU arrays to GPU arrays on AMD GPUs
    A_list_gpu = [ROCArray{T}(A) for A in A_list]
    B_list_gpu = [ROCArray{T}(B) for B in B_list]
    x_list_gpu = [ROCArray{T}(x) for x in x_list]
    x_gpu = [ROCArray{T}(x) for x in x]
    d_list_gpu = [ROCArray{T}(d) for d in d_list]
    return A_list_gpu, B_list_gpu, x_list_gpu, x_gpu, d_list_gpu
end

function construct_block_tridiagonal(A_list, B_list, d_list)

    T = eltype(A_list[1])
    N = length(A_list)
    n = size(A_list[1], 1)
    total_size = N * n  # Total dimension of the sparse matrix

    I = Int[]
    J = Int[]
    V = T[]

    # Fill the sparse matrix
    for i in 1:N
        row_offset = (i - 1) * n
        for j in 1:N
            col_offset = (j - 1) * n
            if i == j
                A = A_list[i]
            elseif j == i + 1
                A = B_list[i]
            elseif j == i - 1
                A = B_list[j]'
            else
                continue  # Skip zero blocks
            end

            # Insert block into sparse matrix
            for row in 1:n
                for col in 1:n
                    push!(I, row_offset + row)
                    push!(J, col_offset + col)
                    push!(V, A[row, col])
                end
            end
        end
    end

    # Construct sparse matrix
    M = sparse(I, J, V, total_size, total_size)

    # Construct right-hand side vector
    d_vec = vcat(d_list...)

    return M, d_vec
end


# function generate_tridiagonal_system(N::Int, n::Int)

#     A_L_list = zeros(n, n, N)
#     for i in 1:N
#         L = LowerTriangular(randn(Float64, n, n))
#         for i in 1:n
#             L[i, i] = abs(L[i, i]) + 1.0
#         end
#         A_L_list[:, :, i] = L
#     end

#     B_L_list = zeros(n, n, N-1)
#     for i in 1:N-1
#         B_L_list[:, :, i] = randn(Float64, n, n)
#     end

#     A_list = zeros(n, n, N)
#     A_list[:, :, 1] = A_L_list[:, :, 1] * A_L_list[:, :, 1]'
#     for i in 2:N
#         A_list[:, :, i] = A_L_list[:, :, i] * A_L_list[:, :, i]' + B_L_list[:, :, i-1]' * B_L_list[:, :, i-1]
#     end

#     # Generate B_list (off-diagonal block matrices)
#     B_list = zeros(n, n, N-1)
#     for i in 1:N-1
#         B_list[:, :, i] = A_L_list[:, :, i] * B_L_list[:, :, i]'
#     end

#     # Generate x_true
#     x_true = rand(N, n)

#     # Compute d_list
#     d_list = zeros(N, n)
#     d_list[1, :] = A_list[:, :, 1] * x_true[1, :] + B_list[:, :, 1] * x_true[2, :]

#     @views for i in 2:N-1
#         d_list[i, :] = B_list[:, :, i-1]' * x_true[i-1, :] + A_list[:, :, i] * x_true[i, :] + B_list[:, :, i] * x_true[i+1, :]
#     end

#     d_list[N, :] = B_list[:, :, N-1]' * x_true[N-1, :] + A_list[:, :, N] * x_true[N, :]

#     # Flatten d_list into a vector
#     d = zeros(N * n)
#     @views for i in 1:N
#         d[(i-1)*n+1:i*n] = d_list[i, :]
#     end

#     # Reshape x_true to a vector
#     x_true = reshape(x_true', N*n)

#     # Construct the block tridiagonal matrix
#     BigMatrix = construct_block_tridiagonal(A_list, B_list)

#     return BigMatrix, d, x_true, A_list, B_list
# end
