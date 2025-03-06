module BlockStructuredSolvers

using LinearAlgebra.LAPACK: potrf!
using LinearAlgebra.BLAS: gemm!, gemv!, trsv!, trsm! #TODO check version of BLAS

export initialize, factorize!, solve!
export initialize_sequential_cholesky_factor, factorize_sequential_cholesky_factor!, solve_sequential_cholesky_factor!

include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")

end