using Test
using BlockStructuredSolvers
import LinearAlgebra: I
using SparseArrays

import CUDA: CuMatrix, DeviceMemory, functional

# Set random seed for deterministic tests
using Random
Random.seed!(42)

# Check if CUDA tests should be run based on command line arguments
const RUN_CUDA_TESTS = "cuda=true" in ARGS

# Skip CUDA tests if not supported or if explicitly disabled
if RUN_CUDA_TESTS && !functional()
    @warn "CUDA is requested for testing but not functional on this system. CUDA tests will be skipped."
    cuda_enabled = false
else
    cuda_enabled = RUN_CUDA_TESTS
end

include("utils.jl")
include("test_cuda.jl")
# include("test_sequential_cholesky_factor.jl")
# include("test_CSR_interface.jl")