using LinearAlgebra, Random

function generate_spd_matrix(k, a=2.0)
    """Generate a random SPD matrix of size k × k."""
    A = randn(k, k)
    A = 0.5 * (A + A')  # Make symmetric
    A += a * I  # Ensure positive definiteness
    return A
end

function generate_block_tridiagonal(n, k)
    """Generate a block tridiagonal SPD matrix of size (nk) × (nk)."""
    D_blocks = [generate_spd_matrix(k, 2.0) for _ in 1:n]
    C_blocks = [randn(k, k) * 0.5 for _ in 1:(n - 1)]
    
    A = zeros(n * k, n * k)
    
    for i in 1:n
        A[(i-1)*k+1:i*k, (i-1)*k+1:i*k] .= D_blocks[i]
        if i < n
            A[(i-1)*k+1:i*k, i*k+1:(i+1)*k] .= C_blocks[i]
            A[i*k+1:(i+1)*k, (i-1)*k+1:i*k] .= C_blocks[i]'  # Symmetric
        end
    end

    return A
end

# Example: Generate a 5-block (each of size 3×3) tridiagonal SPD matrix
n_blocks = 5
block_size = 3
A = generate_block_tridiagonal(n_blocks, block_size)

# Check if the matrix is SPD using Cholesky decomposition
cholesky(A)  # If no error, A is SPD

# Print the resulting matrix
println("Generated SPD Block Tridiagonal Matrix:")
display(A)
