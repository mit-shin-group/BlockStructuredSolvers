module BlockStructuredSolvers

# using LinearAlgebra.LAPACK: potrf!
# using LinearAlgebra.BLAS: gemm!, gemv!, trsv!, trsm! #TODO check version of BLAS

# using CUDA.CUSOLVER: potrf!
# using CUDA.CUBLAS: gemv!, gemm!, trsv!, trsm!

import LinearAlgebra.BLAS: gemm! as lagemm!
import LinearAlgebra.BLAS: gemv! as lagemv!
import LinearAlgebra.BLAS: trsv! as latrsv!
import LinearAlgebra.BLAS: trsm! as latrsm!
import LinearAlgebra.LAPACK: potrf! as lapotrf!

# GPU BLAS/LAPACK operations
import CUDA.CUBLAS: gemm! as cugemm!
import CUDA.CUBLAS: gemv! as cugemv!
import CUDA.CUBLAS: trsv! as cutrsv!
import CUDA.CUBLAS: trsm! as cutrsm!
import CUDA.CUSOLVER: potrf! as cupotrf!

using CUDA

export mypotrf!, mygemv!, mygemm!, mytrsv!, mytrsm!
export cholesky_factorize!, cholesky_solve!
export initialize, factorize!, solve!
export initialize_sequential_cholesky_factor, factorize_sequential_cholesky_factor!, solve_sequential_cholesky_factor!
export initialize_CUDA, factorize_CUDA!, solve_CUDA!

include("myBLAS.jl")
include("backbone.jl")
include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")
include("CUDA_solve.jl")

end