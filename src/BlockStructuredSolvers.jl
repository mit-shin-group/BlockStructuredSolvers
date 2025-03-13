module BlockStructuredSolvers

# CPU BLAS/LAPACK operations
import LinearAlgebra.BLAS: gemm! as lagemm!
import LinearAlgebra.BLAS: trsm! as latrsm!
import LinearAlgebra.LAPACK: potrf! as lapotrf!
import LinearAlgebra: norm as lanorm

# GPU BLAS/LAPACK operations
import CUDA.CUBLAS: gemm! as cugemm!
import CUDA.CUBLAS: trsm! as cutrsm!
import CUDA.CUSOLVER: potrf! as cupotrf!
import CUDA: norm as cunorm

# CUDA types
import CUDA: StridedCuMatrix, CuMatrix, CuArray

# Export functions
export mypotrf!, mygemm!, mytrsm!, mynorm
export cholesky_factorize!, cholesky_solve!
export initialize, factorize!, solve!
export initialize_sequential_cholesky_factor, factorize_sequential_cholesky_factor!, solve_sequential_cholesky_factor!

# Include files
include("myBLAS.jl")
include("backbone.jl")
include("block_cholesky_factor_solve.jl")
include("sequential_cholesky_factor_solve.jl")

end