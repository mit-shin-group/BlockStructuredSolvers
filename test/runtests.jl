using Test
using BlockStructuredSolvers
import LinearAlgebra: I
using SparseArrays

import CUDA: CuMatrix, DeviceMemory

# Set random seed for deterministic tests
using Random
Random.seed!(42)

include("utils.jl")
include("test_cuda.jl")
include("test_rocm.jl")
# include("test_sequential_cholesky_factor.jl")
# include("test_CSR_interface.jl")
