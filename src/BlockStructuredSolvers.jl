module BlockStructuredSolvers

# CPU BLAS/LAPACK operations
import LinearAlgebra.BLAS: gemm! as lagemm!
import LinearAlgebra.BLAS: trsm! as latrsm!
import LinearAlgebra.LAPACK: potrf! as lapotrf!
import LinearAlgebra: norm as lanorm
import LinearAlgebra: I

# GPU BLAS/LAPACK operations #TODO move CUDA as weakdep
import CUDA.CUBLAS: gemm! as cugemm!
import CUDA.CUBLAS: trsm! as cutrsm!
import CUDA.CUSOLVER: potrf! as cupotrf!
import CUDA: norm as cunorm

# CUDA types
import CUDA: StridedCuMatrix, CuMatrix, CuArray, zeros
import CUDA.CUBLAS: gemm_batched!, trsm_batched!, cublasDgemmBatched
import CUDA.CUSOLVER: potrfBatched!, unchecked_cusolverDnDpotrfBatched

# SparseArrays
import SparseArrays: SparseMatrixCSC
using CUDA

# Export functions
export _bss_norm
export BlockTriDiagData, initialize, factorize!, solve!
export copy_vector_of_arrays!, add_vector_of_arrays!, set_zero!, update_boundary_solution!, compute_schur_rhs!, cholesky_factorize!, cholesky_solve!, solve_non_separator_blocks!, cholesky_solve_batched!, cholesky_factorize_batched!

# Include files
include("myBLAS.jl")
include("backbone.jl")
include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")

end