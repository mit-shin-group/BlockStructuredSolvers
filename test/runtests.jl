using Test
using Random
using BlockStructuredSolvers
using SparseArrays
using LinearAlgebra

Random.seed!(42)

include("utils.jl")
include("test_cpu.jl")
include("test_cuda.jl")
include("test_rocm.jl")