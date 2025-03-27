function generate_data(N::Int, n::Int)
    # Generate CPU matrices
    A_list = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        temp = randn(n, n)
        A_list[i] = temp * temp' + n * I
    end
    
    B_list = Vector{Matrix{Float64}}(undef, N-1)
    for i in 1:N-1
        temp = randn(n, n)
        B_list[i] = temp
    end
    
    x_list = Vector{Matrix{Float64}}(undef, N)
    x = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        x_list[i] = rand(n, 1)
        x[i] = zeros(n, 1)
    end
    
    d_list = Vector{Matrix{Float64}}(undef, N)
    d_list[1] = A_list[1] * x_list[1] + B_list[1] * x_list[2]
    @views for i = 2:N-1
        d_list[i] = B_list[i-1]' * x_list[i-1] + A_list[i] * x_list[i] + B_list[i] * x_list[i+1]
    end
    d_list[N] = B_list[N-1]' * x_list[N-1] + A_list[N] * x_list[N]
    
    return A_list, B_list, x_list, x, d_list
end

function construct_block_tridiagonal(A_list, B_list, d_list)
    N = length(A_list)
    n = size(A_list[1], 1)
    total_size = N * n  # Total dimension of the sparse matrix

    I = Int[]
    J = Int[]
    V = Float64[]

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