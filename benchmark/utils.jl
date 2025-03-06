export construct_block_tridiagonal

function construct_block_tridiagonal(A_list, B_list)
    _, n, N = size(A_list)
    blocks = Matrix{Float64}[]

    # Construct the block matrix row-wise
    for i = 1:N
        row_blocks = Any[]
        for j = 1:N
            if i == j
                push!(row_blocks, A_list[:, :, i])  # Diagonal blocks
            elseif j == i + 1
                push!(row_blocks, B_list[:, :, i])  # Upper diagonal blocks
            elseif j == i - 1
                push!(row_blocks, B_list[:, :, j]')  # Lower diagonal blocks (transpose)
            else
                push!(row_blocks, zeros(n, n))  # Zero blocks elsewhere
            end
        end
        push!(blocks, hcat(row_blocks...))
    end

    return vcat(blocks...)
end

function generate_tridiagonal_system(N::Int, n::Int)  #TODO make the whole matrix positive definite
    # Generate A_list (diagonal block matrices)
    A_list = zeros(n, n, N)
    for i in 1:N
        temp = randn(Float64, n, n)
        A_list[:, :, i] = temp * temp' + n * I
    end

    # Generate B_list (off-diagonal block matrices)
    B_list = zeros(n, n, N-1)
    for i in 1:N-1
        temp = randn(Float64, n, n)
        B_list[:, :, i] = temp
    end

    # Generate x_true
    x_true = rand(N, n)

    # Compute d_list
    d_list = zeros(N, n)
    d_list[1, :] = A_list[:, :, 1] * x_true[1, :] + B_list[:, :, 1] * x_true[2, :]

    @views for i in 2:N-1
        d_list[i, :] = B_list[:, :, i-1]' * x_true[i-1, :] + A_list[:, :, i] * x_true[i, :] + B_list[:, :, i] * x_true[i+1, :]
    end

    d_list[N, :] = B_list[:, :, N-1]' * x_true[N-1, :] + A_list[:, :, N] * x_true[N, :]

    # Flatten d_list into a vector
    d = zeros(N * n)
    @views for i in 1:N
        d[(i-1)*n+1:i*n] = d_list[i, :]
    end

    # Reshape x_true to a vector
    x_true = reshape(x_true', N*n)

    # Construct the block tridiagonal matrix
    BigMatrix = construct_block_tridiagonal(A_list, B_list)

    return BigMatrix, d, x_true, A_list, B_list
end
