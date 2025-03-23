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

function detect_block_tridiagonal(A::SparseMatrixCSC)
    n = size(A, 1)
    
    # Get the sparsity pattern
    rows, cols, _ = findnz(A)
    
    # For a block tridiagonal matrix with block size 'b', 
    # we expect dense blocks of size b×b along the main diagonal
    # and first sub/super diagonals
    
    # Find the average distance between "jumps" in the sparsity pattern
    row_jumps = []
    
    # Scan the first few rows to find pattern jumps
    # This works because in block tridiagonal matrices, the sparsity pattern
    # repeats every block_size rows
    prev_cols = Set()
    for row in 1:min(500, n)
        # Get columns with non-zeros for this row
        row_cols = Set(cols[rows .== row])
        
        # If we have previous data to compare
        if !isempty(prev_cols)
            # Calculate how much the pattern changed
            diff = length(symdiff(row_cols, prev_cols))
            
            # If we have a significant change, this might indicate a block boundary
            if diff > length(row_cols) / 2
                push!(row_jumps, row)
            end
        end
        
        prev_cols = row_cols
    end
    
    # Calculate distances between jumps
    if length(row_jumps) >= 2
        jumps_diffs = diff(row_jumps)
        
        # If we have consistent jumps, that's likely our block size
        if !isempty(jumps_diffs) && maximum(jumps_diffs) - minimum(jumps_diffs) < 5
            block_size = round(Int, sum(jumps_diffs) / length(jumps_diffs))
            if n % block_size == 0
                num_blocks = n ÷ block_size
                return num_blocks, block_size
            end
        end
    end
    
    # # If the above approach didn't work, try a simpler method:
    # # For each potential divisor, check if blocks of that size are dense
    
    # # Find all divisors of n
    # divisors = filter(i -> n % i == 0, 1:isqrt(n))
    # append!(divisors, n .÷ divisors)
    # sort!(divisors)
    
    # # Try divisors in descending order (prefer fewer, larger blocks)
    # for block_size in reverse(divisors)
    #     # Skip very small blocks
    #     if block_size < 5
    #         continue
    #     end
        
    #     num_blocks = n ÷ block_size
        
    #     # Check density of the first few diagonal blocks
    #     dense_blocks = true
        
    #     for i in 1:min(3, num_blocks)
    #         block_range = ((i-1)*block_size+1):(i*block_size)
            
    #         # Check diagonal block density
    #         diag_block = A[block_range, block_range]
    #         if nnz(diag_block) < 0.5 * block_size^2
    #             dense_blocks = false
    #             break
    #         end
            
    #         # Check off-diagonal block if not at the end
    #         if i < num_blocks
    #             next_range = (i*block_size+1):((i+1)*block_size)
    #             off_diag_block = A[block_range, next_range]
    #             if nnz(off_diag_block) < 0.1 * block_size^2
    #                 dense_blocks = false
    #                 break
    #             end
    #         end
    #     end
        
    #     if dense_blocks
    #         return num_blocks, block_size
    #     end
    # end
    

    # for block_size in reverse(divisors)
    #     if block_size > 10  # Avoid tiny blocks
    #         num_blocks = n ÷ block_size
    #         return num_blocks, block_size
    #     end
    # end
    
    # # Last resort fallback
    # return n, 1
end