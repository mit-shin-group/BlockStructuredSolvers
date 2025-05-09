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
export BlockTriDiagData_seq, BlockTriDiagData_batched
export initialize_seq, initialize_batched, factorize!, solve!

# Include files
include("myBLAS.jl")
include("backbone_cuda.jl")
include("gpu_seq.jl")
include("gpu_batched.jl")

end