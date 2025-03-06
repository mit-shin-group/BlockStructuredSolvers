module BlockStructuredSolvers

using LinearAlgebra.LAPACK: potrf!
using LinearAlgebra.BLAS: gemm!, gemv!, trsv!, trsm! #TODO check version of BLAS

using CUDA

export initialize, factorize!, solve!
export initialize_sequential_cholesky_factor, factorize_sequential_cholesky_factor!, solve_sequential_cholesky_factor!
export initialize_CUDA, factorize_CUDA!, solve_CUDA!

include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")
include("CUDA_solve.jl")

end