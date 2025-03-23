using Test
using BlockStructuredSolvers

import CUDA: CuMatrix, DeviceMemory

# include("test_block_cholesky_factor.jl")
# include("test_sequential_cholesky_factor.jl")
include("test_CSR_interface.jl")