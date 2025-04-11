using DelimitedFiles

@kwdef mutable struct TBDSolverOptions <: MadNLP.AbstractOptions
    #TODO add ordering
end

mutable struct TBDSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, BlockTriDiagData}
    tril::CUSPARSE.CuSparseMatrixCSC{T}
    
    # not used
    x_gpu::CUDA.CuVector{T}
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
    # N = 50
    # n = 2
    solver = initialize(N, n, eltype(csc), true)
    
    # Create full vector but never used
    x_gpu = CUDA.zeros(T, N*n)
    b_gpu = CUDA.zeros(T, N*n)

    return TBDSolver(solver, csc, x_gpu, b_gpu, opt, logger)
end

function MadNLP.factorize!(solver::TBDSolver{T}) where T

    fill!(solver.inner.LHS_vec, 0)
    fill_block_vecs!(solver.tril, solver.inner.LHS_vec, solver.inner.N, solver.inner.n)
    BlockStructuredSolvers.factorize!(solver.inner)
    return solver
end

function MadNLP.solve!(solver::TBDSolver{T}, d) where T

    fill!(solver.inner.d, 0)
    println("d = ", CUDA.norm(solver.inner.d))
    copyto!(solver.inner.d, d)
    println("d = ", CUDA.norm(solver.inner.d))
    BlockStructuredSolvers.solve!(solver.inner, solver.inner.d_list)
    copyto!(d, view(solver.inner.d, 1:length(d)))

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

function detect_spaces_and_divide_csc(csc_matrix::CUSPARSE.CuSparseMatrixCSC{T}) where T
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

function fill_block_vecs!(
    A_tril_csc::CUSPARSE.CuSparseMatrixCSC{T},
    block_vecs::CuArray{T},
    N::Int,
    n::Int
) where T
    # Convert GPU sparse matrix to CPU for extraction
    A_cpu = SparseMatrixCSC(A_tril_csc)
    S = size(A_cpu, 1)
    
    offset = 0

    for k = 0:N-1
        row_start = k * n + 1
        row_end   = min((k + 1) * n, S)
        row_size  = row_end - row_start + 1
        col_start = row_start

        # Extract and symmetrize A_k (on CPU)
        Ak_cpu = zeros(Float32, n, n)
        
        # For each column in the block
        for col_idx = 1:n
            col = col_start + col_idx - 1
            if col > row_end
                break
            end
            
            # For each row in this column (from the sparse representation)
            for ptr = A_cpu.colptr[col]:A_cpu.colptr[col+1]-1
                row = A_cpu.rowval[ptr]
                val = A_cpu.nzval[ptr]
                
                # If the row is within our current block's range
                if row >= row_start && row <= row_end
                    i = row - row_start + 1
                    j = col_idx
                    
                    # Set both (i,j) and (j,i) for symmetry
                    Ak_cpu[i, j] = val
                    if i != j  # Avoid setting diagonal twice
                        Ak_cpu[j, i] = val
                    end
                end
            end
        end

        # If last diagonal block is smaller, pad with identity
        if row_size < n && k == N-1
            for i = row_size+1:n
                Ak_cpu[i, i] = 1.0f0
            end
        end

        # Transfer complete block to GPU and update block_vecs
        Ak_gpu = CuArray(Ak_cpu)
        block_vecs[offset+1 : offset+n^2] .= reshape(Ak_gpu, :, 1)
        offset += n^2

        # B_k (off-diagonal)
        if k < N - 1
            Bk_cpu = zeros(Float32, n, n)

            r_start = (k+1) * n + 1
            r_end   = min((k+2) * n, S)
            c_start = k * n + 1
            c_end   = min((k+1) * n, S)

            # More explicit iteration for clarity
            for col_idx = 1:n
                col = c_start + col_idx - 1
                if col > c_end
                    break
                end
                
                for ptr = A_cpu.colptr[col]:A_cpu.colptr[col+1]-1
                    row = A_cpu.rowval[ptr]
                    val = A_cpu.nzval[ptr]
                    
                    if row >= r_start && row <= r_end
                        i = col_idx
                        j = row - r_start + 1
                        Bk_cpu[i, j] = val
                    end
                end
            end

            # Transfer complete block to GPU and update block_vecs
            Bk_gpu = CuArray(Bk_cpu)
            block_vecs[offset+1 : offset+n^2] .= reshape(Bk_gpu, :, 1)
            offset += n^2
        end
    end
end