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
    
    BlockStructuredSolvers.solve!(solver.inner, d)
    
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

# # Extracts x_list_gpu from x_gpu using unsafe_wrap
# function extract_x_list!(x_gpu::CUDA.CuVector{T}, x_list_gpu::Vector{CuMatrix{T}}, N::Int, n::Int) where T

#    # Get the raw pointer to the full vector data
#    x_ptr = pointer(x_gpu)
   
#    for i in 1:N
#        x_list_gpu[i] = unsafe_wrap(CuMatrix{T}, x_ptr + n *(i-1) * sizeof(T), (n, 1); 
#                                own=false)
#    end
# end

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