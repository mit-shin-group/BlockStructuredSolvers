function TBDSolver(A::SparseMatrixCSC{T, Int}) where T

    N, n = detect_spaces_and_divide_csc(A)
    solver = initialize(N, n, eltype(A), true)
    extract_AB_list(solver, A)
    
    return solver

end

function factorize!(solver::BlockTriDiagData)

    solver.factorize!(solver)
    
    return solver

end

function solve!(solver::BlockTriDiagData, d)

    #TODO convert d to d_list
    solver.solve!(solver, d)
    #TODO convert d_list to d

    return d

end

function extract_AB_list(solver::BlockTriDiagData, A::SparseMatrixCSC{T, Int}) where T

    A_list = solver.A_list
    B_list = solver.B_list
    n = solver.n
    N = solver.N
   
   # Fill A_list and B_list from the sparse matrix
   for col in 1:size(A, 2)
       block_col = (col - 1) ÷ n + 1
       col_in_block = (col - 1) % n + 1
       
       for ptr in A.colptr[col]:(A.colptr[col+1]-1)
           row = A.rowval[ptr]
           val = A.nzval[ptr]
           
           block_row = (row - 1) ÷ n + 1
           row_in_block = (row - 1) % n + 1
           
           if block_row == block_col
               # Diagonal block
               A_list[block_row][row_in_block, col_in_block] = val
           elseif block_row == block_col - 1
               # Upper off-diagonal block
               B_list[block_row][row_in_block, col_in_block] = val
           elseif block_row == block_col + 1
               # Lower off-diagonal block (transpose of upper)
               B_list[block_col][col_in_block, row_in_block] = val
           end
       end
   end

end

function extract_d_list(d_list, d::Vector{T}, N::Int, n::Int) where T
    
    # Fill d_list from d
    for i in 1:N
        # Calculate the range of indices for this block
        start_idx = (i-1)*n+1
        end_idx = min(i*n, length(d))
        
        # Only copy values if we have data to copy
        if start_idx <= length(d)
            # Copy available values (d_list is already initialized with zeros)
            d_list[i][1:(end_idx-start_idx+1), 1] = view(d, start_idx:end_idx)
        end
    end
    
end

function detect_spaces_and_divide_csc(csc_matrix::SparseMatrixCSC)
    # Extract the necessary components from the CSC matrix
    colptr = csc_matrix.colptr
    rowval = csc_matrix.rowval

    num_rows = size(csc_matrix, 1)
    num_cols = size(csc_matrix, 2)
    
    # We only need to track the first column seen for each row to calculate span
    # Using typemax(Int) as a sentinel value for uninitialized entries
    first_col_seen = fill(typemax(Int), num_rows)
    
    # Track maximum span across all rows
    max_span = 0
    
    @inbounds for col in 1:num_cols
        for ptr in colptr[col]:(colptr[col+1]-1)
            row = rowval[ptr]
            
            # If this is the first time seeing this row, record the column
            if first_col_seen[row] == typemax(Int)
                first_col_seen[row] = col
            end
            
            # Calculate current span and update max_span if needed
            # Current span = current column - first column + 1
            current_span = col - first_col_seen[row] + 1
            max_span = max(max_span, current_span)
        end
    end
    
    # Ensure we had at least some non-zero entries
    if max_span == 0
        return num_rows, 1
    end
    
    # Estimate block size
    result = max_span ÷ 3
    
    # Return number of blocks and block size (ensuring result is at least 1)
    return num_rows ÷ max(1, result), max(1, result)
end