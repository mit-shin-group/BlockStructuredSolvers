module BlockStructuredSolvers

# CPU BLAS/LAPACK operations
import LinearAlgebra.BLAS: gemm! as lagemm!
import LinearAlgebra.BLAS: trsm! as latrsm!
import LinearAlgebra.LAPACK: potrf! as lapotrf!
import LinearAlgebra: norm as lanorm

# GPU BLAS/LAPACK operations #TODO move CUDA as weakdep
import CUDA.CUBLAS: gemm! as cugemm!
import CUDA.CUBLAS: trsm! as cutrsm!
import CUDA.CUSOLVER: potrf! as cupotrf!
import CUDA: norm as cunorm

# CUDA types
import CUDA: StridedCuMatrix, CuMatrix, CuArray
import CUDA.CUBLAS: gemm_batched!, trsm_batched! 
import CUDA.CUSOLVER: potrfBatched!

# SparseArrays
import SparseArrays: SparseMatrixCSC

# Export functions
export _bss_norm
export initialize, factorize!, solve!
export copy_vector_of_arrays!
export construct_block_tridiagonal, generate_data, detect_block_tridiagonal, detect_spaces_and_divide_csc

# Include files
include("myBLAS.jl")
include("backbone.jl")
include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")
include("interface.jl")

end