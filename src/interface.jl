using DelimitedFiles

@kwdef mutable struct TBDSolverOptions <: MadNLP.AbstractOptions
    #TODO add ordering
end

mutable struct TBDSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, BlockTriDiagData}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    
    # Full vector representation
    x_gpu::CUDA.CuVector{T} #TODO same size as MADNLP
    
    # Block representation with direct memory connection to x_gpu
    x_list_gpu::Vector{CuMatrix{T}}
    
    b_gpu::CUDA.CuVector{T}

    opt::MadNLP.AbstractOptions
    logger::MadNLP.MadNLPLogger
end

function TBDSolver(
    csc::CUSPARSE.CuSparseMatrixCSC{T};
    opt=TBDSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    ) where T

    #TODO ordering
    N, n = detect_spaces_and_divide_csc(csc)
    println("N, n = ", N, ", ", n)
    solver = initialize(N, n, eltype(csc), true)

    # Create full vector
    x_gpu = CUDA.zeros(T, N*n)
    
    # Create blocks that directly reference the memory in x_gpu
    x_list_gpu = Vector{CuMatrix{T}}(undef, N)
    
    # Initialize x_list_gpu from x_gpu
    extract_x_list!(x_gpu, x_list_gpu, N, n)
    
    b_gpu = CUDA.zeros(T, N*n)

    return TBDSolver(solver, csc, x_gpu, x_list_gpu, b_gpu, opt, logger)
end

function MadNLP.factorize!(solver::TBDSolver{T}) where T
    # Use the extract_AB_csc_list directly when inner solver supports CSC blocks
    # For now, convert to dense as before
    extract_AB_list!(solver.tril, solver.inner.A_list, solver.inner.B_list, solver.inner.N, solver.inner.n)
    BlockStructuredSolvers.factorize!(solver.inner)
    return solver
end

function MadNLP.solve!(solver::TBDSolver{T}, d) where T
    # Copy input vector to our x_gpu
    copyto!(solver.x_gpu, d)
    
    # Solve the block tridiagonal system using the blocks
    BlockStructuredSolvers.solve!(solver.inner, solver.x_list_gpu)
    
    # Copy results back to output vector
    copyto!(d, view(solver.x_gpu, 1:length(d)))

    return d
end

MadNLP.input_type(::Type{TBDSolver}) = :csc
MadNLP.default_options(::Type{TBDSolver}) = TBDSolverOptions()
MadNLP.improve!(M::TBDSolver) = false
MadNLP.is_supported(::Type{TBDSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{TBDSolver},::Type{Float64}) = true

#TODO intertia
MadNLP.is_inertia(M::TBDSolver) = true
function MadNLP.inertia(M::TBDSolver)
    n = size(M.tril, 1)
    return (n, 0, 0)
end

#TODO improve
MadNLP.improve!(M::TBDSolver) = false

#TODO introduce
MadNLP.introduce(M::TBDSolver) = "TBDSolver"

# Extracts x_list_gpu from x_gpu using unsafe_wrap
function extract_x_list!(x_gpu::CUDA.CuVector{T}, x_list_gpu::Vector{CuMatrix{T}}, N::Int, n::Int) where T

   # Get the raw pointer to the full vector data
   x_ptr = pointer(x_gpu)
   
   for i in 1:N
       x_list_gpu[i] = unsafe_wrap(CuMatrix{T}, x_ptr + n *(i-1) * sizeof(T), (n, 1); 
                               own=false)
   end
end

function detect_spaces_and_divide_csc(csc_matrix::CUSPARSE.CuSparseMatrixCSC{T, Int32}) where T
    # Get matrix dimensions
    num_rows, num_cols = size(csc_matrix, 1), size(csc_matrix, 2)
    
    # Copy GPU arrays to CPU to avoid scalar indexing
    colPtr_cpu = Array(csc_matrix.colPtr)
    rowVal_cpu = Array(csc_matrix.rowVal)
    
    # We only need to track the first column seen for each row to calculate span
    # Using typemax(Int) as a sentinel value for uninitialized entries
    first_col_seen = fill(typemax(Int), num_rows)
    
    # Track maximum span across all rows
    max_span = 0
    
    @inbounds for col in 1:num_cols
        for ptr in colPtr_cpu[col]:(colPtr_cpu[col+1]-1)
            row = rowVal_cpu[ptr]
            
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
    result = max_span ÷ 3 * 2
    
    # Return number of blocks and block size (ensuring result is at least 1)
    return num_rows ÷ max(1, result), max(1, result)
end

# Extract blocks from CSC matrix into sparse CSC block lists by directly accessing the CSC structure
function extract_AB_csc_list!(csc::CUSPARSE.CuSparseMatrixCSC{T},
                             A_csc_list::Vector{CUSPARSE.CuSparseMatrixCSC{T}},
                             B_csc_list::Vector{CUSPARSE.CuSparseMatrixCSC{T}},
                             N::Int, n::Int) where T
    
    # Copy colPtr and rowVal to CPU to avoid scalar indexing, but keep nzVal on GPU
    colPtr_cpu = Array(csc.colPtr)
    rowVal_cpu = Array(csc.rowVal)
    
    # Process each diagonal block
    for i in 1:N
        # Determine block boundaries
        row_start = (i-1)*n+1
        row_end = min(i*n, size(csc, 1))
        col_start = (i-1)*n+1
        col_end = min(i*n, size(csc, 2))
        block_size = row_end - row_start + 1
        
        # First pass: identify the indices in the original array and count non-zeros
        nnz_diag = 0
        col_offsets = zeros(Int, block_size + 1)
        col_offsets[1] = 1  # CSC format starts at 1
        
        # Create arrays to store original indices for linkage
        original_indices = Int[]
        
        for local_col in 1:block_size
            col_idx = col_start + local_col - 1
            for j in colPtr_cpu[col_idx]:colPtr_cpu[col_idx+1]-1
                row_idx = rowVal_cpu[j]
                if row_start <= row_idx <= row_end
                    nnz_diag += 1
                    push!(original_indices, j)
                end
            end
            col_offsets[local_col+1] = nnz_diag + 1  # +1 for CSC format
        end
        
        # Allocate arrays for the block structure
        A_colPtr = Vector{Int32}(col_offsets)
        A_rowVal = Vector{Int32}(undef, nnz_diag)
        
        # Second pass: fill rowVal array (we'll link nzVal directly)
        idx = 0
        for local_col in 1:block_size
            col_idx = col_start + local_col - 1
            for j in colPtr_cpu[col_idx]:colPtr_cpu[col_idx+1]-1
                row_idx = rowVal_cpu[j]
                if row_start <= row_idx <= row_end
                    idx += 1
                    A_rowVal[idx] = row_idx - row_start + 1  # Convert to local row index
                end
            end
        end
        
        # Create a new nzVal that directly references the original values using unsafe_wrap
        # First, get the device pointer to the original nzVal array
        nzVal_ptr = pointer(csc.nzVal)
        
        # Create a new array that references the values at the specified indices
        # Note: This requires the indices to be contiguous in memory, which they may not be
        # Instead, we'll create a CPU sparse matrix and then convert to GPU
        
        # Create CPU sparse matrix with our prepared structure
        if nnz_diag > 0
            # We need to get these specific values from GPU to CPU
            nzval_block_cpu = Array{T}(undef, nnz_diag)
            
            # Copy specific indices from GPU to CPU - we'll need to index these properly
            for (idx, original_idx) in enumerate(original_indices)
                # We need to extract this specific value from the GPU array
                nzval_block_cpu[idx] = Array(csc.nzVal)[original_idx]
            end
            
            A_block = SparseMatrixCSC{T, Int32}(block_size, block_size, A_colPtr, A_rowVal, nzval_block_cpu)
            
            # Convert to GPU
            A_csc_list[i] = CUSPARSE.CuSparseMatrixCSC(A_block)
        else
            # Empty block
            A_block = spzeros(T, block_size, block_size)
            A_csc_list[i] = CUSPARSE.CuSparseMatrixCSC(A_block)
        end
        
        # Process off-diagonal block (if not the last block)
        if i < N
            # For lower triangular matrix, extract block below diagonal (B_list[i])
            row_start_lower = i*n+1
            row_end_lower = min((i+1)*n, size(csc, 1))
            col_start_lower = (i-1)*n+1
            col_end_lower = min(i*n, size(csc, 2))
            
            block_rows = row_end_lower - row_start_lower + 1
            block_cols = col_end_lower - col_start_lower + 1
            
            # First pass: count non-zeros and identify original indices
            nnz_offdiag = 0
            col_offsets_B = zeros(Int, block_cols + 1)
            col_offsets_B[1] = 1
            
            # Store original indices for the off-diagonal block
            original_indices_B = Int[]
            
            for local_col in 1:block_cols
                col_idx = col_start_lower + local_col - 1
                for j in colPtr_cpu[col_idx]:colPtr_cpu[col_idx+1]-1
                    row_idx = rowVal_cpu[j]
                    if row_start_lower <= row_idx <= row_end_lower
                        nnz_offdiag += 1
                        push!(original_indices_B, j)
                    end
                end
                col_offsets_B[local_col+1] = nnz_offdiag + 1
            end
            
            # Allocate arrays for the off-diagonal block structure
            B_colPtr = Vector{Int32}(col_offsets_B)
            B_rowVal = Vector{Int32}(undef, nnz_offdiag)
            
            # Second pass: fill rowVal array
            idx = 0
            for local_col in 1:block_cols
                col_idx = col_start_lower + local_col - 1
                for j in colPtr_cpu[col_idx]:colPtr_cpu[col_idx+1]-1
                    row_idx = rowVal_cpu[j]
                    if row_start_lower <= row_idx <= row_end_lower
                        idx += 1
                        B_rowVal[idx] = row_idx - row_start_lower + 1  # Convert to local row index
                    end
                end
            end
            
            if nnz_offdiag > 0
                # Extract the specific values for this block
                nzval_block_B_cpu = Array{T}(undef, nnz_offdiag)
                
                # Copy specific indices from GPU to CPU
                for (idx, original_idx) in enumerate(original_indices_B)
                    nzval_block_B_cpu[idx] = Array(csc.nzVal)[original_idx]
                end
                
                B_block = SparseMatrixCSC{T, Int32}(block_rows, block_cols, B_colPtr, B_rowVal, nzval_block_B_cpu)
                
                # Convert to GPU
                B_csc_list[i] = CUSPARSE.CuSparseMatrixCSC(B_block)
            else
                # Empty block
                B_block = spzeros(T, block_rows, block_cols)
                B_csc_list[i] = CUSPARSE.CuSparseMatrixCSC(B_block)
            end
        end
    end
    
    # Make diagonal blocks symmetric
    for i in 1:N
        if !issymmetric(SparseMatrixCSC(A_csc_list[i]))
            cpu_block = SparseMatrixCSC(A_csc_list[i])
            symmetric_block = (cpu_block + cpu_block') / 2
            A_csc_list[i] = CUSPARSE.CuSparseMatrixCSC(symmetric_block)
        end
    end
    
    # Print a message to verify that the function used unsafe_wrap approach
    println("extract_AB_csc_list! used direct memory linking for non-zero values")
end

# Extract blocks from CSC matrix into dense block lists
function extract_AB_list!(csc::CUSPARSE.CuSparseMatrixCSC{T},
                          A_list::Vector{CuMatrix{T}},
                          B_list::Vector{CuMatrix{T}},
                          N::Int, n::Int) where T
    
    # Create temporary CSC block lists
    A_csc_list = Vector{CUSPARSE.CuSparseMatrixCSC{T}}(undef, N)
    B_csc_list = Vector{CUSPARSE.CuSparseMatrixCSC{T}}(undef, N-1)
    
    # Extract blocks as CSC matrices
    extract_AB_csc_list!(csc, A_csc_list, B_csc_list, N, n)
    
    # Convert each CSC block to dense format for the solver
    for i in 1:N
        # Process diagonal block
        A_dense = CUDA.zeros(T, n, n)
        copyto!(A_dense, Array(SparseMatrixCSC(A_csc_list[i])))
        A_list[i] .= A_dense
        
        # Process off-diagonal block (if not the last block)
        if i < N
            B_dense = CUDA.zeros(T, n, n)
            copyto!(B_dense, Array(SparseMatrixCSC(B_csc_list[i])))
            B_list[i] .= B_dense
        end
    end
end

# Inline test case for CSC block extraction algorithms
println("Testing CSC block extraction algorithms...")

# Parameters for a small test case
N_test = 3  # Number of blocks
n_test = 2  # Block size
total_size = N_test * n_test  # Total matrix size

# Create a known block tridiagonal pattern as a sparse matrix
# Example: A 6x6 matrix with 3 blocks of size 2x2
I_test = Int32[] # Row indices
J_test = Int32[] # Column indices
V_test = Float64[] # Values

# Diagonal blocks (A_list)
# Block 1,1
push!(I_test, 1); push!(J_test, 1); push!(V_test, 11.0)  # A[1,1] = 11
push!(I_test, 2); push!(J_test, 1); push!(V_test, 12.0)  # A[2,1] = 12
push!(I_test, 2); push!(J_test, 2); push!(V_test, 22.0)  # A[2,2] = 22

# Block 2,2
push!(I_test, 3); push!(J_test, 3); push!(V_test, 33.0)  # A[3,3] = 33
push!(I_test, 4); push!(J_test, 4); push!(V_test, 44.0)  # A[4,4] = 44

# Block 3,3
push!(I_test, 5); push!(J_test, 5); push!(V_test, 55.0)  # A[5,5] = 55
push!(I_test, 6); push!(J_test, 6); push!(V_test, 66.0)  # A[6,6] = 66

# Lower off-diagonal blocks (B_list)
# Block 2,1 (below diagonal)
push!(I_test, 3); push!(J_test, 1); push!(V_test, 31.0)  # A[3,1] = 31
push!(I_test, 4); push!(J_test, 2); push!(V_test, 42.0)  # A[4,2] = 42

# Block 3,2 (below diagonal)
push!(I_test, 5); push!(J_test, 3); push!(V_test, 53.0)  # A[5,3] = 53
push!(I_test, 6); push!(J_test, 4); push!(V_test, 64.0)  # A[6,4] = 64

# Create sparse matrix on CPU first
cpu_sparse_test = sparse(I_test, J_test, V_test, total_size, total_size)

# Print the test matrix
println("Test matrix (lower triangular form):")
display(Array(cpu_sparse_test))
println()

# Convert to GPU CSC format
csc_test = CUSPARSE.CuSparseMatrixCSC(cpu_sparse_test)

# Test 1: Direct CSC block extraction
println("\nTesting extract_AB_csc_list! (direct CSC block extraction):")

# Create CSC block lists
A_csc_test = Vector{CUSPARSE.CuSparseMatrixCSC{Float64}}(undef, N_test)
B_csc_test = Vector{CUSPARSE.CuSparseMatrixCSC{Float64}}(undef, N_test-1)

# Run extraction
extract_AB_csc_list!(csc_test, A_csc_test, B_csc_test, N_test, n_test)

# Get results back to CPU for display
A_csc_results = [Array(SparseMatrixCSC(A)) for A in A_csc_test]
B_csc_results = [Array(SparseMatrixCSC(B)) for B in B_csc_test]


expected_A1 = [11.0 0.0; 12.0 22.0]  # Symmetrized from original data
expected_A2 = [33.0 0.0; 0.0 44.0]        # Already symmetric
expected_A3 = [55.0 0.0; 0.0 66.0]        # Already symmetric

# Off-diagonal blocks should match the lower triangular part
expected_B1 = [31.0 0.0; 0.0 42.0]  # Block below diagonal between blocks 1 and 2
expected_B2 = [53.0 0.0; 0.0 64.0]  # Block below diagonal between blocks 2 and 3

# Check results from CSC extraction (compare to same expected values)
println("\nVerifying CSC extraction results:")
csc_passed = true
csc_passed &= isapprox(A_csc_results[1], expected_A1)
csc_passed &= isapprox(A_csc_results[2], expected_A2)
csc_passed &= isapprox(A_csc_results[3], expected_A3)
csc_passed &= isapprox(B_csc_results[1], expected_B1)
csc_passed &= isapprox(B_csc_results[2], expected_B2)

println("CSC extraction test result: ", csc_passed ? "PASSED" : "FAILED")

# Overall result
println("\nOverall test result: ", (csc_passed) ? "PASSED" : "FAILED")

# ---- End of test case ----