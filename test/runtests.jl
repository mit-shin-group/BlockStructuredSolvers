using Test
using BlockStructuredSolvers
using LinearAlgebra
using CUDA

include("test_block_cholesky_factor.jl")
include("test_sequential_cholesky_factor.jl")
# include("test_CUDA.jl")