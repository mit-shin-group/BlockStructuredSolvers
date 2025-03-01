module BlockStructuredSolvers

using LinearAlgebra: ldiv!, mul!, cholesky!, Hermitian
using LinearAlgebra.LAPACK: trtrs!
using LinearAlgebra.BLAS: gemm!

export BlockStructuredData, initialize, factorize!, solve!
export BlockStructuredData_full_cholesky_factor, initialize_full_cholesky_factor, factorize_full_cholesky_factor!, solve_full_cholesky_factor!

include("block_cholesky_factor_solve.jl")
include("full_cholesky_factor_solve.jl")


end