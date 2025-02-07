using LinearAlgebra

function generate_pd_block_tridiagonal(n::Int, block_size::Int)
    blocks = [randn(block_size, block_size) for _ in 1:n]
    
    # Make diagonal blocks positive definite
    for i in 1:n
        blocks[i] = blocks[i] * blocks[i]' + block_size * I
    end

    T = zeros(n * block_size, n * block_size)

    # Assign diagonal blocks
    for i in 1:n
        T[(i-1)*block_size+1:i*block_size, (i-1)*block_size+1:i*block_size] = blocks[i]
    end

    # Assign off-diagonal blocks
    for i in 1:n-1
        B = randn(block_size, block_size)
        T[(i-1)*block_size+1:i*block_size, i*block_size+1:(i+1)*block_size] = B
        T[i*block_size+1:(i+1)*block_size, (i-1)*block_size+1:i*block_size] = B'
    end

    return T
end

# Example usage
n = 5         # Number of blocks
block_size = 3 # Size of each block
T = generate_pd_block_tridiagonal(n, block_size)

println("Generated positive definite block tridiagonal matrix:")
display(T)

# Check if it's positive definite
println("Eigenvalues:", eigvals(T))
println("Is positive definite?", all(eigvals(T) .> 0))
