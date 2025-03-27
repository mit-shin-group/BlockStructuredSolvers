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