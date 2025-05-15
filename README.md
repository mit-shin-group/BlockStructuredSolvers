# BlockStructuredSolvers.jl

A high-performance Julia package for solving block-structured linear systems with a focus on block tridiagonal matrices.

[![Build Status](https://github.com/username/BlockStructuredSolvers.jl/workflows/CI/badge.svg)](https://github.com/username/BlockStructuredSolvers.jl/actions)
[![Coverage](https://codecov.io/gh/username/BlockStructuredSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/username/BlockStructuredSolvers.jl)

## Overview

BlockStructuredSolvers.jl provides specialized solvers for block-structured linear systems with a focus on block tridiagonal matrices. It offers multiple backends for different hardware architectures:

- CPU: Sequential solver for standard CPUs
- CUDA: High-performance GPU solvers using NVIDIA CUDA
- ROCm: GPU solvers for AMD GPUs using ROCm

Each backend supports both single (Float32) and double (Float64) precision computations, allowing you to balance speed and accuracy based on your application needs.

## Features

- **Multiple Backends**: CPU, CUDA (NVIDIA), and ROCm (AMD) support
- **Precision Options**: Both Float32 and Float64 support
- **Algorithm Variants**:
  - Sequential solver: Traditional block tridiagonal algorithm
  - Batched solver: Optimized for parallel execution on GPUs
- **High Performance**: Optimized implementations for each backend
- **Simple Interface**: Consistent API across all backends

## Installation

```julia
using Pkg
Pkg.add("BlockStructuredSolvers")
```

## Quick Start

Here's a simple example of how to use BlockStructuredSolvers to solve a block tridiagonal system:

```julia
using BlockStructuredSolvers
using LinearAlgebra
using Random

# Problem size
N = 100  # Number of blocks
n = 32   # Size of each block

# Choose precision
T = Float64

# CPU solver
function solve_on_cpu()
    # Initialize the solver
    data = initialize_cpu(N, n, T)
    
    # Create and set up matrices
    A_tensor = Array{T, 3}(zeros(n, n, N))
    B_tensor = Array{T, 3}(zeros(n, n, N-1))
    
    # Fill with your data (example: random positive definite matrices)
    for i in 1:N
        temp = randn(T, n, n)
        A_tensor[:, :, i] = temp * temp' + n * I
    end
    
    for i in 1:N-1
        temp = randn(T, n, n)
        B_tensor[:, :, i] = temp
    end
    
    # Copy data to solver
    for i in 1:N
        copyto!(data.A_list[i], A_tensor[:, :, i])
    end
    
    for i in 1:N-1
        copyto!(data.B_list[i], B_tensor[:, :, i])
    end
    
    # Create right-hand side
    d_list = Vector{Matrix{T}}(undef, N)
    for i in 1:N
        d_list[i] = rand(T, n, 1)
    end
    
    # Factorize
    factorize!(data)
    
    # Solve
    solve!(data, d_list)
    
    # Solution is now in d_list
    return d_list
end

# GPU solver using CUDA
function solve_on_gpu()
    using CUDA
    
    # Initialize the batched solver
    data = initialize_batched(N, n, T, CuArray)
    
    # Create and set up matrices
    A_tensor = CuArray{T, 3}(zeros(n, n, N))
    B_tensor = CuArray{T, 3}(zeros(n, n, N-1))
    
    # Fill with your data (example: random positive definite matrices)
    CUDA.@allowscalar for i in 1:N
        temp = randn(T, n, n)
        A_tensor[:, :, i] .= CuArray{T, 2}(temp * temp' + n * I)
    end
    
    CUDA.@allowscalar for i in 1:N-1
        temp = randn(T, n, n)
        B_tensor[:, :, i] .= CuArray{T, 2}(temp)
    end
    
    # Copy data to solver
    copyto!(data.A_tensor, A_tensor)
    copyto!(data.B_tensor, B_tensor)
    
    # Create right-hand side
    d_tensor = CuArray{T, 3}(zeros(T, n, 1, N))
    CUDA.@allowscalar for i in 1:N
        d_tensor[:, :, i] = CuArray{T, 2}(rand(T, n, 1))
    end
    
    # Copy right-hand side to solver
    copyto!(data.d_tensor, d_tensor)
    
    # Factorize
    CUDA.@sync factorize!(data)
    
    # Solve
    CUDA.@sync solve!(data)
    
    # Solution is now in data.d_list
    return data.d_list
end
```

## API Reference

### CPU Solver

```julia
# Initialize a CPU solver
data = initialize_cpu(N, n, T=Float64)

# Factorize the matrix
factorize!(data)

# Solve the system
solve!(data, d_list)
```

### CUDA Solver

```julia
# Initialize a batched CUDA solver
data = initialize_batched(N, n, T=Float64, M=CuArray)

# Initialize a sequential CUDA solver
data = initialize_seq(N, n, T=Float64, M=CuArray)

# Factorize the matrix
CUDA.@sync factorize!(data)

# Solve the system
CUDA.@sync solve!(data)
```

### ROCm Solver

```julia
# Initialize a batched ROCm solver
data = initialize_batched(N, n, T=Float64, M=ROCArray)

# Initialize a sequential ROCm solver
data = initialize_seq(N, n, T=Float64, M=ROCArray)

# Factorize the matrix
AMDGPU.@sync factorize!(data)

# Solve the system
AMDGPU.@sync solve!(data)
```

## Performance

BlockStructuredSolvers.jl is designed for high performance across different hardware architectures. Here are some key performance considerations:

- **GPU vs CPU**: For large systems, GPU implementations can offer significant speedups over CPU implementations.
- **Batched vs Sequential**: The batched solver is generally faster on GPUs for large problems.
- **Float32 vs Float64**: Single precision (Float32) is approximately twice as fast as double precision (Float64) but offers less numerical accuracy.

## Algorithm

The package implements a block Cholesky factorization for block tridiagonal matrices. For a block tridiagonal matrix with diagonal blocks A_i and off-diagonal blocks B_i, the algorithm:

1. Factorizes the diagonal blocks using Cholesky decomposition
2. Updates the remaining blocks using forward and backward substitution
3. Solves the system using the factorized blocks

For the batched implementation, the algorithm is reorganized to maximize parallel execution on GPUs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This package is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use BlockStructuredSolvers.jl in your research, please cite:

```
@software{BlockStructuredSolvers,
  author = {David Jin, Alexis Montoison, and Sungho Shin},
  title = {BlockStructuredSolvers.jl: A High-Performance Julia Package for Solving Block-Structured Linear Systems},
  year = {2024},
  url = {https://github.com/username/BlockStructuredSolvers.jl}
}
```
