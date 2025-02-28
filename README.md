# BlockStructuredSolvers.jl

![CI](https://github.com/mit-shin-group/BlockStructuredSolvers/actions/workflows/ci.yml/badge.svg)

## Overview
`BlockStructuredSolvers.jl` is a Julia package for efficiently solving block tridiagonal systems using Cholesky factorization. The package is designed for structured linear algebra problems that arise in scientific computing and optimization.

## Features
- **Efficient Cholesky factorization** for block tridiagonal matrices
- **Hierarchical solver** using recursive elimination
- **Supports large-scale problems** with optimized BLAS and LAPACK operations
- **Compatible with Julia's built-in linear algebra functions**

## Installation
To install the package, clone the repository and add it to your Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/BlockStructuredSolvers.jl")
```

Alternatively, if developing locally:
```julia
Pkg.develop(path="/path/to/BlockStructuredSolvers")
```

## Usage
### Importing the Package
```julia
using BlockStructuredSolvers
```

### Initializing a Block Tridiagonal System
```julia
N = ...  # Number of blocks
m = ...  # Number of sub-blocks
n = ...  # Size of each block
P = ...  # Number of separators
level = ...  # Hierarchical levels

A_list = rand(N, n, n)   # Diagonal matrices
B_list = rand(N-1, n, n) # Off-diagonal matrices

data = initialize(N, m, n, P, A_list, B_list, level)
```

### Factorizing the System
```julia
factorize(data)
```

### Solving the System
```julia
d = ... # Right-hand side vector
x = zeros(N * n) # Solution vector

solve(data, d, x)
```

## API Reference
### `initialize(N, m, n, P, A_list, B_list, level) -> BlockStructuredSolversData`
Creates and initializes a hierarchical block tridiagonal system.

### `factorize(data::BlockStructuredSolversData)`
Performs Cholesky factorization on the system.

### `solve(data::BlockStructuredSolversData, d, x)`
Solves the block tridiagonal system given a right-hand side `d` and stores the solution in `x`.

## Performance Considerations
- This package leverages **BLAS** and **LAPACK** for optimized linear algebra operations.
- Large systems can benefit from **multi-level elimination** using hierarchical solvers.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests on GitHub.

## License
TODO

## Acknowledgments
TODO
